import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# -*-*-*-Separate BRSet images into a balanced dataset-*-*-*-
# got the following images proportion: 
#             Adequate | Inadequate
# Train         1315   |    1426 
# Test          1413   |    0213 
# Validation    0316   |    0347 

# count the images balance on adequate/inadequate over total
def contBalance(df):
    contAdequate = len(df[df['quality_label'] == 1])
    contInadequate = len(df[df['quality_label'] == 0])
    print("Adequados: {} / Inadequados: {} / Balanceamento adequadas: {:.2f}".format(
        contAdequate, contInadequate, contAdequate / (contAdequate + contInadequate)))

# create classes directories, in case they don't exist
def create_directories(base_path):
    os.makedirs(os.path.join(base_path, '0'), exist_ok=True)
    os.makedirs(os.path.join(base_path, '1'), exist_ok=True)

# move images to directories
def move_images(img_path, dest_path, df):
    create_directories(dest_path)
    for index, row in df.iterrows():
        image_name = f"{row['image_id']}.jpg"
        quality = row['quality_label']
        original_image_path = os.path.join(img_path, image_name)
        destination_path = os.path.join(dest_path, str(quality), image_name)

        if os.path.exists(original_image_path):
            shutil.move(original_image_path, destination_path)
        else:
            print(f"A imagem {image_name} não foi encontrada.")

    print("Processo de movimentação concluído.")

def main():
    image_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos'
    train_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos/train'
    test_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos/test'
    val_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos/validation'

    labels = pd.read_csv('/home/rodrigocm/scratch/datasets/brset/labels.csv', sep=',')
    labels['quality_label'] = labels['quality'].apply(lambda x: 1 if x == 'Adequate' else 0)

    # split data into train, test and validation
    train_df, test_df = train_test_split(labels, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # balance classes on the training set
    train_df_balanced = pd.concat([
        resample(train_df[train_df['quality_label'] == 1], replace=True, n_samples=len(train_df[train_df['quality_label'] == 0]), random_state=42),
        train_df[train_df['quality_label'] == 0]
    ])

    # balance classes on the validation set
    val_df_balanced = pd.concat([
        resample(val_df[val_df['quality_label'] == 1], replace=True, n_samples=len(val_df[val_df['quality_label'] == 0]), random_state=42),
        val_df[val_df['quality_label'] == 0]
    ])

    print("Treino balanceado:")
    contBalance(train_df_balanced)
    print("Validação balanceada:")
    contBalance(val_df_balanced)
    print("Teste:")
    contBalance(test_df)

    # move images
    move_images(img_path=image_path, dest_path=train_path, df=train_df_balanced)
    move_images(img_path=image_path, dest_path=test_path, df=test_df)
    move_images(img_path=image_path, dest_path=val_path, df=val_df_balanced)

main()