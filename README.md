## Clickbait detection

This is a toy project for detecting clickbait images (my original intent), or really any type of image you might want to filter for. It's using [MobileNetV3](https://pytorch.org/vision/main/models/mobilenetv3.html), so I'm hoping it will be small enough to embed in a browser extension, with the end goal being something which screens out certain images with decent accuracy.

### Implemenation goals

#### Big picture

- [x] - Query YouTube for image thumbnails
- [x] - Scrape thumbnail images
- [x] - Tool for hand classifying scraped images
- [x] - Train model using the classified images
- [ ] - Create Chrome/Firefox extension template to identify images

#### Todo

This is my running todo list.

- [x] - Exporting of model after training
- [ ] - Args for saving to file other than _output.sqlite_
- [ ] - Use existing model to help speed up the process of image classification
- [ ] - Args for **train** script that set dropout
- [ ] - Args for **train** script that set weight loss
- [ ] - Args for **train** script that allow loads of existing weights (to continue training)
- [ ] - CLI tool for testing individual images with an exported model

### The setup

There are four scripts which are used to query the api, download the images, classify them and then train the model. They are all works in progress, but if you have ideas feel free to make a pull request.

To get started clone the repo and then create a virtualenv in the root folder and set the PYTHONPATH:

```bash
virtualenv -p python3 venv
. venv/bin/activate
```

Alternatively, you can use [Poetry](https://python-poetry.org/) or something else, as well. Either way, after that, you can install from the **requirements.txt**.

```bash
pip install -r requirements.txt
```

Lastly, in in the **app** folder, change the **template.env** to **.env** and then add in your YouTubeAPI key. This you can get through the [Google's developer console](https://console.cloud.google.com/products/solutions/).

### How to use

Note: I've left my database here, which has everything ticked as being downloaded, but you obviously won't have those files. You can either clear the database and start from scratch of set all the downloaded fields to 0 and rerun.

Also, I've hardcoded **output.sqlite** as the database name in some scripts. I may go back and make this optional later on.

---

The first script queries the YouTubeAPI by using keywords or phrases from the designated text file. Since I've been focused on clickbait you can see there are queries like "Top 10 mysteries of the world" or things like that.

Running from the root folder:

```bash
python -m app.query.main -k keywords.txt
```

I have been using ChatGPT or something similar to generate potential queries. The keywords you put in the file will be removed one by one as they are queried.

Next, you'll scrape the images, which will end up in the **downloads** folder. I've set the default delay to 3 seconds, but you can probably get away with shorter.

```bash
python -m app.scrape.main
```

After you've scraped the images and the field in the database will be set to **1** and you'll want to classify the images. The script used to classify is relatively simple. It's a tkinter window that accepts any keyboard input. Whatever you press becomes that _class_ for that photo.

To run the classifier do the following and then start hitting keys. In my database you'll see 1 and 2, representing clickbait and not clickbait, respectively.

```bash
python -m app.classify.main
```

Lastly, you'll want to train the model. You can run this to train without saving the model:

```bash
python -m app.train.main
```

Or to export the model use:

```bash
python app/train/main.py -e out-model
```

As noted above, this is a work in progress, so there's nothing setup to run the final model independently of training.
