# Massive Data Institute Project
## Background
The major goal of this project is to quantify discrimination patterns in customer service of four airline companies. The whole process can be summarized as data collection, cleaning data, feature enginnering, exploratory data analysis, machine learning deployment. The first section below document the ETL process and data governance. The cache/plots folder lists the visualization results. 

## Modules

- filter_names.py --> filter out non-human names, and remain the twitter accounts named by human names
- race&gender.py --> get gender info from first name using NLTK library, gender_guesser library, bio description info, get race info from first name using ethnicolr library
- sentiment analysis.py --> get sentiments from reply text(need to upgrade versions)
- unknown_gender_bug.py --> fix bugs of rows of unknown gender that have first name and last name
- customer_reply.py --> word frequency distribution patterns of customer replies and identify patterns of different representatives
- customer_reply_2.py --> the relationship among response time, public_impression_count and user_follower_count
- web scraping.py ---> download images online
- image classification.py -->extract genders and race from images used ViT and ResNet

## Output
- cache/output --> document the csv output(not present on Github for data privacy)
- cache/plots --> document the interactive visualization results