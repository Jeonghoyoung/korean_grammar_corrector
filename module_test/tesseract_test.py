import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import utils.file_util as ft


file = '../data/pdf/test2/pf_test2.pdf'
s = '../data/pdf/test2/jpg'

pdf_pages = convert_from_path(file)
img_list = []

for i, page in enumerate(pdf_pages):
    file_name = f'{s}/pf_test2_{i}'
    page.save(f'{file_name}.jpg', 'JPEG')
    img_list.append(file_name)


result=[]
flist = os.listdir(s)
flist.sort()
for file in flist:
    print(file)
    text = pytesseract.image_to_string(Image.open(f'{s}/{file}'))
    text_list = text.split('\n')
    ft.write_list_file(text_list,f'../data/pdf/test2/txt/{file.split(".jpg")[0]}.txt')
    result.extend(text_list)
# print(result)