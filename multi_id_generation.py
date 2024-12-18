import os
import argparse
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from uniportrait import inversion
from uniportrait.uniportrait_attention_processor import attn_args
from uniportrait.uniportrait_pipeline import UniPortraitPipeline

def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    ret = cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return ret, (left, top)

def process_faceid_image(pil_faceid_image, face_app):
    np_faceid_image = np.array(pil_faceid_image.convert("RGB"))
    img = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)
    faces = face_app.get(img)  # bgr
    if len(faces) == 0:
        # padding, try again
        _h, _w = img.shape[:2]
        _img, left_top_coord = pad_np_bgr_image(img)
        faces = face_app.get(_img)
        if len(faces) == 0:
            print("Warning: No face detected in the image. Continue processing...")

        min_coord = np.array([0, 0])
        max_coord = np.array([_w, _h])
        sub_coord = np.array([left_top_coord[0], left_top_coord[1]])
        for face in faces:
            face.bbox = np.minimum(np.maximum(face.bbox.reshape(-1, 2) - sub_coord, min_coord), max_coord).reshape(4)
            face.kps = face.kps - sub_coord

    faces = sorted(faces, key=lambda x: abs((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])), reverse=True)
    if not faces:
        raise ValueError("No face detected in the image.")

    faceid_face = faces[0]
    norm_face = face_align.norm_crop(img, landmark=faceid_face.kps, image_size=224)
    pil_faceid_align_image = Image.fromarray(cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB))

    return pil_faceid_align_image

def prepare_single_faceid_cond_kwargs(pil_faceid_image=None, pil_faceid_supp_images=None,
                                      pil_faceid_mix_images=None, mix_scales=None, face_app=None):
    pil_faceid_align_images = []
    if pil_faceid_image:
        pil_faceid_align_images.append(process_faceid_image(pil_faceid_image, face_app))
    if pil_faceid_supp_images and len(pil_faceid_supp_images) > 0:
        for supp_image_path in pil_faceid_supp_images:
            pil_faceid_supp_image = Image.open(supp_image_path)
            pil_faceid_align_images.append(process_faceid_image(pil_faceid_supp_image, face_app))

    mix_refs = []
    mix_ref_scales = []
    if pil_faceid_mix_images:
        for mix_image_path, mix_scale in zip(pil_faceid_mix_images, mix_scales):
            if mix_image_path:
                pil_faceid_mix_image = Image.open(mix_image_path)
                mix_refs.append(process_faceid_image(pil_faceid_mix_image, face_app))
                mix_ref_scales.append(mix_scale)

    single_faceid_cond_kwargs = None
    if len(pil_faceid_align_images) > 0:
        single_faceid_cond_kwargs = {
            "refs": pil_faceid_align_images
        }
        if len(mix_refs) > 0:
            single_faceid_cond_kwargs["mix_refs"] = mix_refs
            single_faceid_cond_kwargs["mix_scales"] = mix_ref_scales

    return single_faceid_cond_kwargs

def text_to_multi_id_generation_process(
        pil_faceid_image_1_path=None, pil_faceid_supp_images_1_paths=None,
        pil_faceid_mix_image_1_1_path=None, mix_scale_1_1=0.0,
        pil_faceid_mix_image_1_2_path=None, mix_scale_1_2=0.0,
        pil_faceid_image_2_path=None, pil_faceid_supp_images_2_paths=None,
        pil_faceid_mix_image_2_1_path=None, mix_scale_2_1=0.0,
        pil_faceid_mix_image_2_2_path=None, mix_scale_2_2=0.0,
        faceid_scale=0.0, face_structure_scale=0.0,
        prompt="", negative_prompt="",
        num_samples=1, seed=-1,
        image_resolution="512x512",
        inference_steps=25,
        output_dir="output"
):
    if seed == -1:
        seed = None

    # Initialize FaceAnalysis
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=["detection"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Prepare condition kwargs for first ID
    faceid_cond_kwargs_1 = prepare_single_faceid_cond_kwargs(
        pil_faceid_image=Image.open(pil_faceid_image_1_path) if pil_faceid_image_1_path else None,
        pil_faceid_supp_images=pil_faceid_supp_images_1_paths,
        pil_faceid_mix_images=[pil_faceid_mix_image_1_1_path, pil_faceid_mix_image_1_2_path],
        mix_scales=[mix_scale_1_1, mix_scale_1_2],
        face_app=face_app
    )

    # Prepare condition kwargs for second ID
    faceid_cond_kwargs_2 = prepare_single_faceid_cond_kwargs(
        pil_faceid_image=Image.open(pil_faceid_image_2_path) if pil_faceid_image_2_path else None,
        pil_faceid_supp_images=pil_faceid_supp_images_2_paths,
        pil_faceid_mix_images=[pil_faceid_mix_image_2_1_path, pil_faceid_mix_image_2_2_path],
        mix_scales=[mix_scale_2_1, mix_scale_2_2],
        face_app=face_app
    )

    cond_faceids = []
    if faceid_cond_kwargs_1 is not None:
        cond_faceids.append(faceid_cond_kwargs_1)
    if faceid_cond_kwargs_2 is not None:
        cond_faceids.append(faceid_cond_kwargs_2)

    # Reset attn args
    attn_args.reset()
    # Set faceid condition
    attn_args.lora_scale = 1.0 if len(cond_faceids) == 1 else 0.0  # single-faceid lora
    attn_args.multi_id_lora_scale = 1.0 if len(cond_faceids) > 1 else 0.0  # multi-faceid lora
    attn_args.faceid_scale = faceid_scale if len(cond_faceids) > 0 else 0.0
    attn_args.num_faceids = len(cond_faceids)
    print(attn_args)

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    base_model_path = "models/SG161222/Realistic_Vision_V5.1_noVAE"
    vae_model_path = "models/stabilityai/sd-vae-ft-mse"
    controlnet_pose_ckpt = "models/lllyasviel/control_v11p_sd15_openpose"
    image_encoder_path = "models/IP-Adapter/models/image_encoder"
    ip_ckpt = "models/IP-Adapter/models/ip-adapter_sd15.bin"
    face_backbone_ckpt = "models/glint360k_curricular_face_r101_backbone.bin"
    uniportrait_faceid_ckpt = "models/uniportrait-faceid_sd15.bin"
    uniportrait_router_ckpt = "models/uniportrait-router_sd15.bin"

    # Load controlnet
    pose_controlnet = ControlNetModel.from_pretrained(controlnet_pose_ckpt, torch_dtype=torch_dtype)

    # Load SD pipeline
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch_dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=[pose_controlnet],
        torch_dtype=torch_dtype,
        scheduler=noise_scheduler,
        vae=vae,
    ).to(device)

    # Load uniportrait pipeline
    uniportrait_pipeline = UniPortraitPipeline(
        pipe, image_encoder_path, ip_ckpt=ip_ckpt,
        face_backbone_ckpt=face_backbone_ckpt,
        uniportrait_faceid_ckpt=uniportrait_faceid_ckpt,
        uniportrait_router_ckpt=uniportrait_router_ckpt,
        device=device, torch_dtype=torch_dtype
    )

    h, w = map(int, image_resolution.split("x"))
    prompt_list = [prompt] * num_samples
    negative_prompt_list = [negative_prompt] * num_samples
    images = uniportrait_pipeline.generate(
        prompt=prompt_list,
        negative_prompt=negative_prompt_list,
        cond_faceids=cond_faceids,
        face_structure_scale=face_structure_scale,
        seed=seed,
        guidance_scale=7.5,
        num_inference_steps=inference_steps,
        image=[torch.zeros([1, 3, h, w])],
        controlnet_conditioning_scale=[0.0]
    )

    final_out = []
    for pil_image in images:
        final_out.append(pil_image)

    for single_faceid_cond_kwargs in cond_faceids:
        final_out.extend(single_faceid_cond_kwargs["refs"])
        if "mix_refs" in single_faceid_cond_kwargs:
            final_out.extend(single_faceid_cond_kwargs["mix_refs"])

    # Save output images
    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(final_out):
        img_path = os.path.join(output_dir, f"output_{idx+1}.png")
        img.save(img_path)
        print(f"Saved output image: {img_path}")

def main():
    parser = argparse.ArgumentParser(description="Text-to-Multi-ID Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="nsfw", help="Negative prompt")
    parser.add_argument("--pil_faceid_image_1", type=str, required=True, help="Path to first ID image")
    parser.add_argument("--pil_faceid_supp_images_1", type=str, nargs='*', default=[], help="Paths to additional first ID images")
    parser.add_argument("--pil_faceid_mix_image_1_1", type=str, default=None, help="Path to Mix ID 1 image for first ID")
    parser.add_argument("--mix_scale_1_1", type=float, default=0.0, help="Mix scale 1 for first ID")
    parser.add_argument("--pil_faceid_mix_image_1_2", type=str, default=None, help="Path to Mix ID 2 image for first ID")
    parser.add_argument("--mix_scale_1_2", type=float, default=0.0, help="Mix scale 2 for first ID")
    parser.add_argument("--pil_faceid_image_2", type=str, required=True, help="Path to second ID image")
    parser.add_argument("--pil_faceid_supp_images_2", type=str, nargs='*', default=[], help="Paths to additional second ID images")
    parser.add_argument("--pil_faceid_mix_image_2_1", type=str, default=None, help="Path to Mix ID 1 image for second ID")
    parser.add_argument("--mix_scale_2_1", type=float, default=0.0, help="Mix scale 1 for second ID")
    parser.add_argument("--pil_faceid_mix_image_2_2", type=str, default=None, help="Path to Mix ID 2 image for second ID")
    parser.add_argument("--mix_scale_2_2", type=float, default=0.0, help="Mix scale 2 for second ID")
    parser.add_argument("--faceid_scale", type=float, default=0.7, help="Face ID Scale")
    parser.add_argument("--face_structure_scale", type=float, default=0.3, help="Face Structure Scale")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--image_resolution", type=str, default="512x512", help="Image resolution (HxW)")
    parser.add_argument("--inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output images")

    args = parser.parse_args()

    text_to_multi_id_generation_process(
        pil_faceid_image_1_path=args.pil_faceid_image_1,
        pil_faceid_supp_images_1_paths=args.pil_faceid_supp_images_1,
        pil_faceid_mix_image_1_1_path=args.pil_faceid_mix_image_1_1,
        mix_scale_1_1=args.mix_scale_1_1,
        pil_faceid_mix_image_1_2_path=args.pil_faceid_mix_image_1_2,
        mix_scale_1_2=args.mix_scale_1_2,
        pil_faceid_image_2_path=args.pil_faceid_image_2,
        pil_faceid_supp_images_2_paths=args.pil_faceid_supp_images_2,
        pil_faceid_mix_image_2_1_path=args.pil_faceid_mix_image_2_1,
        mix_scale_2_1=args.mix_scale_2_1,
        pil_faceid_mix_image_2_2_path=args.pil_faceid_mix_image_2_2,
        mix_scale_2_2=args.mix_scale_2_2,
        faceid_scale=args.faceid_scale,
        face_structure_scale=args.face_structure_scale,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_samples=args.num_samples,
        seed=args.seed,
        image_resolution=args.image_resolution,
        inference_steps=args.inference_steps,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
