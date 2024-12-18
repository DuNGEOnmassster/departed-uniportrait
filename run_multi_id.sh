python multi_id_generation.py \
    --prompt "Two people are playing baseball on the glass." \
    --pil_faceid_image_1 data_ip/bengio.png \
    --mix_scale_1_1 0.5 \
    --mix_scale_1_2 0.3 \
    --pil_faceid_image_2 data_ip/hinton.png \
    --mix_scale_2_1 0.6 \
    --mix_scale_2_2 0.4 \
    --faceid_scale 0.7 \
    --face_structure_scale 0.3 \
    --num_samples 2 \
    --seed 42 \
    --image_resolution "512x512" \
    --inference_steps 25 \
    --output_dir "outputs/multi_id"