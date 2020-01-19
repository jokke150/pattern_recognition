import os
import twint
from twint.storage import write
import json

SARCASTIC_TWEETS_FILE = "../sarcastic.json" 
NORMAL_TWEETS_FILE = "../non-sarcastic.json" 

def get_distinct_usernames(file):
	with open(SARCASTIC_TWEETS_FILE, encoding='UTF-8') as f:
	  	usernames = set()
	  	json_object = json.load(f)

	for line in json_object:
	    usernames.add(line['username'].strip())

	return usernames

def get_tweets_by_user(usernames):
	tweets_by_user = {}

	for username in usernames:
		tweets = []

		c = twint.Config()
		c.Limit=20
		c.Username = username
		c.Store_object = True
		c.Store_object_tweets_list = tweets
		c.Debug = False
		c.Hide_output = True
		twint.run.Search(c)

		tweets_by_user[username] = tweets

	return tweets_by_user


if __name__=="__main__":
	# Read usernames and make them distinct
	usernames = get_distinct_usernames(SARCASTIC_TWEETS_FILE)
	usernames.update(get_distinct_usernames(NORMAL_TWEETS_FILE))
	usernames = list(usernames)

	# exclude already scraped usernames
	scraped_usernames = []
	with open('tweets_by_user.json', "r") as f:
		scraped_usernames = json.load(f).keys()
	usernames = [x for x in usernames if x not in scraped_usernames]

	print("usernames to scrape: " + str(len(usernames))) 

	write_threshold = 50 # number of users to fetch tweets from before writing
	iterations = len(usernames) // write_threshold + 1
	for i in range(iterations):
		usernames_slice = usernames[i * write_threshold: (i + 1) * write_threshold]
		tweets_by_user = get_tweets_by_user(usernames_slice)

		with open('tweets_by_user.json', "a", newline='', encoding="utf-8") as f:
			# Remove last }
			if i > 0 or len(scraped_usernames) > 0:
				f.seek(0, os.SEEK_END)
				f.seek(f.tell() - 1, os.SEEK_SET)
				f.truncate()

			json_tweets_by_user = {}

			for username, tweets in tweets_by_user.items():
				json_tweets = []
				
				for tweet in tweets:
					null, data = write.struct(tweet, False, "tweet")
					json_tweet = json.dumps(data, ensure_ascii=False)
					json_tweets.append(json_tweet)

				json_tweets_by_user[username] = json_tweets

			json_tweets_slice = json.dumps(json_tweets_by_user)

			if i > 0 or len(scraped_usernames) > 0:
				json_tweets_slice = json_tweets_slice.replace('{', ', ', 1)
			f.write(json_tweets_slice)

		print("Stored tweets for users: " + str(usernames_slice))





		


