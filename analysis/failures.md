# Failure Analysis (CLIP zero-shot on Food-101)

Settings:
- Model: openai/clip-vit-base-patch32
- Prompt: a close-up photo of {label}
- Samples scanned: 2000
- Failures saved: 10

## Summary of common failure patterns
The majority of observed failures involve the beignets class, which highlights both cultural and visual ambiguity in the dataset. In many regions (including parts of Canada such as Montréal), the term “beigne” is commonly used to refer to donuts, creating an inherent semantic overlap between beignets and donuts. This linguistic ambiguity likely contributes to confusion in visually similar fried dough dishes.

Additional failure patterns are driven by presentation context. Churros are often photographed alongside dipping sauces (e.g., caramel or chocolate), which closely resemble how the beignets in this study are served. In several cases, the presence of sauces or additional items in the frame caused the model to focus on contextual cues rather than the food itself.

Less frequent failures stem from unusual plating or shape, where beignets are presented in forms that resemble other foods such as takoyaki or eggs benedict. These results suggest that CLIP relies heavily on global visual patterns and contextual similarity rather than fine-grained culinary distinctions.

## Failure examples

| file          | true label | predicted label | why this might have happened | category |
|---------------|------------|-----------------|------------------------------|----------|
| [failure_0.png](images/failure_0.png) | beignets   | falafel         | The shape, and color of the falafel confused the ai into thinking it was a beignets                             |     Similar-looking foods     |
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