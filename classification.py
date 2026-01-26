import os
import clip
import torch
import pandas as pd
from PIL import Image

def run_split_indoor_outdoor(
    root_dir: str,
    classes_file: str,
    prompt_template: str = "this photo was taken {}",
    model_name: str = "ViT-B/32",
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device)

    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    text_inputs = torch.cat([clip.tokenize(prompt_template.format(c))
        for c in classes]).to(device)
    
    rows = []
    for dir_name, _, file_list in os.walk(root_dir):
        for image_name in file_list:
            image_file = os.path.join(dir_name, image_name)
            image = Image.open(image_file)
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)

            # Print results
            print(f'\n{image_file} Top predictions:\n')
            for value, index in zip(values, indices):
                print(f"{classes[index]:>10s}: {100 * value.item():.2f}%")
            
            if (classes[indices[0]] == "outdoor"):
                destination_folder = "results\\outdoor"
            else:
                destination_folder = "results\\indoor"

            rows.append({
                "image": image_name,
                "I/O": "outdoor" if classes[indices[0]] == "outdoor" else "indoor",
                "light/meteo": None,
                "place": None,
                "objects": []
            })

            image.save(os.path.join(destination_folder, image_name))
    return pd.DataFrame(rows)

def run_clip_classification(
    root_dir: str,
    classes_file: str,
    prompt_template: str,
    model_name: str = "ViT-B/32",
    top_k: int = 5,
    device: str = None,
    df_column: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device)

    # Load classes
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    # Build prompts
    text_inputs = torch.cat([
        clip.tokenize(prompt_template.format(c))
        for c in classes
    ]).to(device)

    # Precompute text features once
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    for dir_name, _, file_list in os.walk(root_dir):
        for image_name in file_list:
            image_path = os.path.join(dir_name, image_name)

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
                continue

            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            values, indices = similarity[0].topk(top_k)

            print(f"\n{image_path} Top predictions:")
            for value, index in zip(values, indices):
                print(f"{classes[index]:>15s}: {100 * value.item():.2f}%")
            if (df_column):
                if (df_column == "objects"):
                    pass
                else:
                    df.at[image_name, df_column] = classes[indices[0]]
    return df

df = run_split_indoor_outdoor(
    root_dir="..\\testDataSet",
    classes_file="classes/classes_io.txt"
)

df = df.set_index("image")

print("-------------------INDOOR------------------- \n")
#INDOOR
#places
print("PLACES \n")
df = run_clip_classification(
    root_dir="results/indoor",
    classes_file="classes/classes_i_places.txt",
    prompt_template="this photo was taken in a {}",
    df_column="place"
)
#objects
print("OBJECTS \n")
df = run_clip_classification(
    root_dir="results/indoor",
    classes_file="classes/classes_i_objects.txt",
    prompt_template="in this photo, there is at least one {}",
    top_k=4,
    df_column="objects"
)
#light
print("LIGHT \n")
df = run_clip_classification(
    root_dir="results/indoor",
    classes_file="classes/classes_i_light.txt",
    prompt_template="in this photo, the level of light is {}",
    df_column="light/meteo"
)

print("-------------------OUTDOOR------------------- \n")
#OUTDOOR
#places
print("PLACES \n")
df = run_clip_classification(
    root_dir="results/outdoor",
    classes_file="classes/classes_o_places.txt",
    prompt_template="this photo was taken in a {}",
    df_column="place"
)
#objects
print("OBJECTS \n")
df = run_clip_classification(
    root_dir="results/outdoor",
    classes_file="classes/classes_o_objects.txt",
    prompt_template="in this photo, there is at least one {}",
    df_column="objects"
)
#meteo
print("METEO \n")
df = run_clip_classification(
    root_dir="results/outdoor",
    classes_file="classes/classes_o_meteo.txt",
    prompt_template="in this photo, it is {}",
    df_column="light/meteo"
)

print(df)
