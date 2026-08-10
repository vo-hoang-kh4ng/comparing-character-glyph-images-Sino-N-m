import numpy as np
import pandas as pd
import os
import cv2
from sklearn.metrics import pairwise

# 1. Read excel file to get info
df_quocngu_sinonm = pd.read_excel('QuocNgu_SinoNom_Dic.xlsx')
df_final_char = pd.read_excel('final_characteristics-v2.xlsx')

# 2. Enter input
# input_text = 'trăm'
# test_image_path = './test_images/tram.jpg'

# input_text = 'năm'
# test_image_path = './test_images/nam.jpg'

input_text = 'ta'
test_image_path = './test_images/ta.jpg'

# input_text = 'trong'
# test_image_path = './test_images/trong.jpg'

print(f"Image of text: {input_text}")

# 3. Get SinoNom list from input_text
sino_nom_list = df_quocngu_sinonm[df_quocngu_sinonm['QuocNgu'] == input_text]['SinoNom'].tolist()

# 4. Mapping SinoNom to CHAR and get UNICODE
df_unicode_mapping = df_final_char[df_final_char['CHAR'].isin(sino_nom_list)]

# 5. Feature extraction
def extract_histogram(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Gaussian blur filter
    edges = cv2.Canny(image, 100, 200)  # Edge detection with Canny filter

    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contrast_stretched = cv2.normalize(binary, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    hist = cv2.calcHist([contrast_stretched], [0], None, [256], [0, 256])  # Histogram calculation
    hist = cv2.normalize(hist, hist).flatten()  # Histogram normalization

    return hist

test_histogram = extract_histogram(test_image_path)

# 7. Calculate similarity among test image & UNICODE.jpg images in ./images folder
similarity_results = []

for _, row in df_unicode_mapping.iterrows():
    unicode = row['UNICODE']
    sino_nom = row['CHAR']
    image_path = f'./images/{unicode}.jpg'

    if os.path.exists(image_path):
        image_histogram = extract_histogram(image_path)
        
        distance = np.linalg.norm(test_histogram - image_histogram)
        similarity = 1 / (1 + distance)

        similarity_results.append((unicode, sino_nom, similarity))

# 8. Sort & get top 10 similarities
sorted_results = sorted(similarity_results, key=lambda x: x[2], reverse=True)[:10]

# 9. Create DataFrame output with UNICODE, SinoNom, and similarity score
output_df = pd.DataFrame(sorted_results, columns=['UNICODE', 'SinoNom', 'Similarity'])

print(output_df)

# 10. Save Excel/CSV file
output_df.to_excel('./output/output_histogram_similarity.xlsx', index=False)
output_df.to_csv('./output/output_histogram_similarity.csv', index=False)