import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def cont_balance(dataframe):
    adequate_count = 0
    inadequate_count = 0
    for item in dataframe['quality_label']:
        if item == 1:
            adequate_count += 1
        else:
            inadequate_count += 1
    total = adequate_count + inadequate_count

    return (adequate_count/total)

def process_csv_data():
    path = '/home/rodrigocm/scratch/datasets/eyeq/labels_eyeq.csv'

    # if data_df['images'][i] == 0, image is good, otherwise it is bad
    data_df = pd.read_csv(path, sep=',')
    data_df['quality_label'] = data_df['quality'].apply(lambda x: 1 if x == 0 else 0)

    return data_df

# create classes directories, in case they don't exist
def create_directories(base_path):
    os.makedirs(os.path.join(base_path, '0'), exist_ok=True)
    os.makedirs(os.path.join(base_path, '1'), exist_ok=True)

def move_images(dataframe, img_path, target_path):
    create_directories(target_path)
    for index, row in dataframe.iterrows():
        image_name = f"{row['image']}"
        quality = row['quality_label']
        original_image_path = os.path.join(img_path, image_name)
        destination_path = os.path.join(target_path, str(quality), image_name)

        if os.path.exists(original_image_path):
            shutil.move(original_image_path, destination_path)
        else:
            print(f"A imagem {image_name} não foi encontrada.")

    print("Processo de movimentação concluído.")

def main():
    images_path = '/home/rodrigocm/scratch/datasets/eyeq/images'
    train_path = '/home/rodrigocm/scratch/datasets/eyeq/images/train'
    test_path = '/home/rodrigocm/scratch/datasets/eyeq/images/test'
    val_path = '/home/rodrigocm/scratch/datasets/eyeq/images/validation'

    data_df = process_csv_data()

    # 72% for training, 10% for testing, 18% for validation
    train_df, test_df = train_test_split(data_df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)


    # check how balanced the data is, may need to balance close to 50/50 later
    print("Adequadas / Total (treino): {}".format(cont_balance(train_df)))
    print("Adequadas / Total (teste): {}".format(cont_balance(test_df)))
    print("Adequadas / Total (validação): {}".format(cont_balance(val_df)))

    # move images to their respective directories
    move_images(train_df, images_path, train_path)
    move_images(test_df, images_path, test_path)
    move_images(val_df, images_path, val_path)

main()  