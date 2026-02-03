# Failure Analysis (CLIP zero-shot on Food-101)

Settings:
- Model: openai/clip-vit-base-patch32
- Prompt: a close-up photo of {label}
- Samples scanned: 2000
- Failures saved: 10

## Summary of common failure patterns
- Beignets are also known as donuts in certain provinces or places such a montreal. We call donuts "beignes", which brings so much confusion as to what a donut, and a beignets is. Another common failure pattern is observed by the dipping sauces/presentation. Churros are known, to be served on the side with dulce de leche, caramel etc etc. So when we see the fried dough served with those sauces it brings to confusion

## Failure examples

| file          | true label | predicted label | why this might have happened | category |
|---------------|------------|-----------------|------------------------------|----------|
| failure_0.png | beignets   | falafel         | The shape, and color of the falafel confused the ai into thinking it was a beignets                             |     Similar-looking foods     |
| failure_1.png | beignets   |donuts           |    Beignets are often square fried doughs, where as donuts are more circular.                          |    Similar-looking foods      |
| failure_2.png | beignets   | churros         |  The dipping sauces resemble the sauces used on churros. |    Multiple foods in frame      |
| failure_3.png | beignets   | bread_pudding   |   Bread pudding is often served with ice cream therefore confusion  | Multiple foods in frame         |
| failure_4.png | beignets   |donuts           |Beignets are often square fried doughs, where as donuts are more circular.                              |Similar-looking foods          |
| failure_5.png | beignets   | churros         | The dipping sauces resemble the sauces used on churros.                             | Multiple foods in frame          |
| failure_6.png | beignets   | takoyaki        | The unusual shapes and presentation resembles that of takoyaki.  | Unusual presentation / style  Similar-looking foods        |
| failure_7.png | beignets   | churros         | The dipping sauces resemble the sauces used on churros. |    Multiple foods in frame      |
| failure_8.png | beignets   | churros         |The dipping sauces resemble the sauces used on churros. |    Multiple foods in frame      |
| failure_9.png | beignets   | eggs_benedict   | The beignets are placed just like how egg benedicts would be presented.                            |   Unusual presentation / style       |


### Category legend
- Similar-looking foods
- Cropped / blurry / bad angle
- Multiple foods in frame
- Unusual presentation / style
- Label noise (image doesn’t match label)