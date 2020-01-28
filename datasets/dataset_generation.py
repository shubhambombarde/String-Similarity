import os

writeFileObj = open('hotel_reviews.txt', 'w')

for dirname in os.listdir('hotel_reviews'):
  for filename in os.listdir("hotel_reviews/"+str(dirname)):
    print("hotel_reviews/"+str(dirname)+"/"+str(filename))
    readFileObj = open("hotel_reviews/"+str(dirname)+"/"+str(filename), mode='r', encoding='utf-8', errors='ignore')
    # print(readFileObj.readlines())
    # break
    writeFileObj.write(readFileObj.readlines()[0])
    readFileObj.close()
writeFileObj.close()
print('DONE')