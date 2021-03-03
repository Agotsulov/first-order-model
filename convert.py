import matplotlib
matplotlib.use('Agg')
import sys
import yaml
from argparse import ArgumentParser

import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector

import coremltools as ct

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("-o", "--onnx", dest="onnx", action="store_true", help="add convert to onnx.")
    parser.add_argument("-ts", "--torchscript", dest="torchscript", action="store_true", help="add convert to torchscript.")
    parser.add_argument("-q", "--quantize", dest="quantize", action="store_true", help="add dynamic quantize.")
 
    opt = parser.parse_args()


    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=True)



    if opt.quantize: 
        kp_detector = torch.quantization.quantize_dynamic(
            kp_detector, {torch.nn.Linear}, dtype=torch.qint8)

        generator = torch.quantization.quantize_dynamic(
            generator, {torch.nn.Linear}, dtype=torch.qint8)
    
    example_input_kp_detector = torch.randn(1, 3, 256, 256, requires_grad=True)
    
    if opt.torchscript:
        print("CONVERT KP DETECTOR TO TORCHSCRIPT")
        traced_kp_detector = torch.jit.trace(kp_detector, example_input_kp_detector, strict=False)
        traced_kp_detector.save("kp_detector.pt")


    example_input_generator = torch.randn(1, 3, 256, 256, requires_grad=True)
    
    example_input_kp_norm_value = torch.randn(1, 10, 2, requires_grad=True)
    example_input_kp_norm_jacobian = torch.randn(1, 10, 2, 2, requires_grad=True)

    example_input_kp_source_value = torch.randn(1, 10, 2, requires_grad=True)
    example_input_kp_source_jacobian = torch.randn(1, 10, 2, 2, requires_grad=True)
    

    if opt.torchscript:
        print("CONVERT GENERATOR TO TORCHSCRIPT")
        traced_generator = torch.jit.trace(
            generator, (
                example_input_generator,
                example_input_kp_source_value,
                example_input_kp_source_jacobian,
                example_input_kp_norm_value,
                example_input_kp_norm_jacobian,
            ),
            strict=False
        )
        traced_generator.save("generator.pt")
    
    
    if opt.onnx:
        print("CONVERT KP DETECTOR TO ONNX")
        torch.onnx.export(kp_detector,
                    example_input_kp_detector,
                    "vox-kp_detector.onnx",
                    export_params=True,
                    opset_version=11)  

        print("CONVERT GENERATOR TO ONNX")
        torch.onnx.export(generator,
                    (
                        example_input_generator,
                        example_input_kp_source_value,
                        example_input_kp_source_jacobian,
                        example_input_kp_norm_value,
                        example_input_kp_norm_jacobian,
                    ), "vox-generator.onnx",
                    export_params=True,
                    opset_version=11)  



    # TODO: Fix torch.inverse and torch.F.grid_sample
    # print("CONVERT KP DETECTOR TO COREML")
    # model = ct.convert(
    #     traced_kp_detector,
    #     inputs=[ct.ImageType(name="input_1", shape=example_input_kp_detector.shape)], 
    # )
    # model.save("kp_detector.mlmodel")

    # print("CONVERT GENERATOR TO COREML")
    # model = ct.convert(
    #     traced_generator,
    #     inputs=[
    #         ct.ImageType(name="input_1", shape=example_input_generator.shape),
    #         ct.TensorType(name="input_2", shape=example_input_kp_source_value.shape),
    #         ct.TensorType(name="input_3", shape=example_input_kp_source_jacobian.shape),
    #         ct.TensorType(name="input_4", shape=example_input_kp_norm_value.shape),
    #         ct.TensorType(name="input_5", shape=example_input_kp_norm_jacobian.shape),
    #     ], 
    # )
    # model.save("generator.mlmodel")

    