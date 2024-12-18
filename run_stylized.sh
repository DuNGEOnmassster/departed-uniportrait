python stylized_generation.py \
    --pil_ip_image data_ip/trump.png \
    --pil_faceid_image Paintings/Vangogh-1.png \
    --mix_scale_1 0.5 \
    --mix_scale_2 0.3 \
    --faceid_scale 0.7 \
    --face_structure_scale 0.3 \
    --ip_scale 0.7 \
    --num_samples 2 \
    --seed 42 \
    --image_resolution "768x512" \
    --inference_steps 25 \
    --output_dir "outputs/stylized"
