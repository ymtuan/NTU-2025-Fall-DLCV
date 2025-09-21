import os
import torch
from tqdm import tqdm
from dataset.deep_globe import DeepGlobe, classToRGB, is_image_file
from helper import create_model_load_weights, Evaluator, collate_test

# ----------------- Settings -----------------
n_class = 7
mode = 3  # 1: global only, 2: local from global, 3: global from local
batch_size = 6
sub_batch_size = 6
size_g = 508
size_p = 508

data_path = "../../../data_2025/p2_data/"
model_path = "./saved_models/"
path_g = "fpn_deepglobe_global.epoch129.pth"
path_g2l = "fpn_deepglobe_global2local.epoch28.pth"
path_l2g = "fpn_deepglobe_local2global.epoch28.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- Dataset -----------------
ids_test = [img_name for img_name in os.listdir(os.path.join(data_path, "validation"))
            if img_name.endswith("_sat.jpg") and is_image_file(img_name)]

dataset_test = DeepGlobe(os.path.join(data_path, "validation"), ids_test, label=False)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                              batch_size=batch_size,
                                              num_workers=os.cpu_count(),
                                              collate_fn=collate_test,
                                              shuffle=False,
                                              pin_memory=True)

# ----------------- Model -----------------
model, global_fixed = create_model_load_weights(n_class, mode, evaluation=True,
                                                path_g=os.path.join(model_path, path_g),
                                                path_g2l=os.path.join(model_path, path_g2l),
                                                path_l2g=os.path.join(model_path, path_l2g))
model.to(device)
model.eval()
global_fixed = global_fixed.to(device) if global_fixed is not None else None

# ----------------- Evaluator -----------------
evaluator = Evaluator(n_class, size_g=(size_g, size_g),
                      size_p=(size_p, size_p),
                      sub_batch_size=sub_batch_size,
                      mode=mode,
                      test=True)

# ----------------- Prediction -----------------
if not os.path.isdir("./prediction/"):
    os.mkdir("./prediction/")

with torch.no_grad():
    for sample_batched in tqdm(dataloader_test, desc="Predicting"):
        # Pass PIL images directly (do NOT convert to tensor)
        images = sample_batched['image']
        ids = sample_batched['id']

        # Evaluate batch
        eval_input = {'image': images, 'id': ids}
        predictions, predictions_global, predictions_local = evaluator.eval_test(eval_input, model, global_fixed)

        # Save predictions
        for i in range(len(images)):
            if mode == 1:
                pred_img = classToRGB(predictions_global[i])
            else:
                pred_img = classToRGB(predictions[i])
            from torchvision import transforms
            transforms.functional.to_pil_image(pred_img * 255.).save(f"./prediction/{ids[i]}_mask.png")

print("Predictions saved in ./prediction/")