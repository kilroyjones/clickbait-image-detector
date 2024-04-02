## Clickbait detection

This project is a toy project for detecting clickbait images (my original intent), or really any type of image you might want to filter for. It's using [MobileNetV3](https://pytorch.org/vision/main/models/mobilenetv3.html), so I'm hoping it will be small enough to embed in a browser extension, with the end goal being something which screen out certain images with decent accuracy.

### Implemenation goals

- [x] - Query YouTube for image thumbnails
- [x] - Scrape thumbnail images
- [x] - Tool for hand classifying scraped images
- [x] - Train model using the classified images
- [ ] - Use existing model to help speed up the process of image classification
- [ ] - Create Chrome/Firefox extension template to identify images

### How to use

There are four scripts which combine to make
