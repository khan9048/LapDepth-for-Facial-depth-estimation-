from PIL import Image
image = Image.open(r'out_my_examples_1/out_fk_ali.jpg')
new_image = image.resize((640, 480))
new_image.save(r'out_my_examples_1/out_fk_ali_1.png')
print(image.size) # Output: (1200, 776)
print(new_image.size) # Output: (400, 400)
