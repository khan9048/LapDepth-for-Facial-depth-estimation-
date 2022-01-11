from PIL import Image
image = Image.open(r'my_examples_1/out_my_examples_1/out_my_examples_1/fk_ali.jpg')
new_image = image.resize((640, 480))
new_image.save(r'my_examples_1/out_my_examples_1/out_my_examples_1/fk_ali_11.jpg')
print(image.size) # Output: (1200, 776)
print(new_image.size) # Output: (400, 400)
