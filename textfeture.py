import os

print("= Install library")
os.system("pip install -Uqq h5py numpy pandas matplotlib bs4 regex flair tables fastai ")


print("= Import library")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

from bs4 import BeautifulSoup

import fastai.text.core

import nltk
import sklearn

import string

# !conda install -y -c conda-forge spacy-model-en_core_web_sm

import spacy

if ('check_en_core_web_sm' not in locals()):
	spacy.cli.download("en_core_web_sm")
	check_en_core_web_sm = True
import en_core_web_sm
import re

import flair

#Check we we using GPU
if flair.torch.cuda.is_available():
	print("GPU ON")
	flair.device = flair.torch.device('cuda:0')
else:
	print("cpu mode")
	flair.device = flair.torch.device('cpu')



def clean_text_before_save(text, twitter_user=True, url2URL=True):
	text = text.strip()

	text = BeautifulSoup(text, 'html.parser').get_text()

	# test = test.strip(); # remove all leading and tailing spaces
	# end of line to space
	if (url2URL):
		text = re.sub(r'https?:\/\/[\w\.\/]+', 'URL', text)
	if (twitter_user):
		text = re.sub("@\w+", "USER", text)

	text = text.replace("=>", " at ")

	# replace "word's" with "word is"
	text = re.sub(r" (\w+)'s ", " \\1 is ", text, 0, re.MULTILINE)
	# "Add spaces around / and #"
	text = re.sub(r'([/#\\])', " \\1 ", text, 0, re.MULTILINE)

	text = fastai.text.core.fix_html(text)
	text = fastai.text.core.replace_all_caps(text)
	text = fastai.text.core.spec_add_spaces(text)
	text = fastai.text.core.rm_useless_spaces(text)

	# Words with punctuations and special characters
	for p in string.punctuation:
		text = text.replace(p, f' {p} ')

	text = re.sub("\s+", " ", text)

	text.replace("\n+", " ")
	text.replace("\r+", " ")
	return text


def get_text_and_more(soup):
	html_img_src = ""
	for tag in soup.find_all(True):
		if (tag.name not in ['style', 'noscript', 'script', 'input', 'head', 'title', 'meta', '[document]']):
			# print(tag.name)
			if tag.string is not None:
				# print(">> ",tag.name," ",tag.string)
				if (tag.name == 'a'):
					if (tag.get('href') is None):
						html_img_src += "na "
					else:
						html_img_src += "HREF " + tag.get('href') + " "
				elif (tag.name == 'img'):
					if (tag.get('src') is None):
						html_img_src += "na "
					else:
						html_img_src += "IMAGE " + tag.get('src') + " "
					if (tag.get('alt') is None):
						html_img_src += "na "
					else:
						html_img_src += "ALT " + tag.get('alt') + " "
				else:
					html_img_src += tag.string + " "
	return html_img_src


def get_text_from_html(contents, debug=0):
	result = {}
	soup = BeautifulSoup(contents, 'html.parser')
	soup.prettify()
	result['text'] = soup.get_text()
	result['text_strip'] = soup.get_text(strip=True)
	result['clean_text'] = clean_text_before_save(result['text_strip'])
	title = soup.find('title')
	if (title is not None):
		result['title'] = clean_text_before_save(title.get_text("|", strip=True))
	else:
		result['title'] = ""

	result["tag_" + "sum"] = 0
	for x in ['iframe', 'span', 'link', 'select', 'hr', 'li', 'p', 'div', 'ul', 'td', 'tr', 'img', 'a', 'submit',
			  'href', "article", "h1", "h2", "h3", "h4", "h5", "source", "video"]:
		result["tag_" + x] = len(soup.find_all(x))
		result["tag_" + "sum"] += result["tag_" + x]

	# Each sentance is new row
	result['stripped_strings'] = []
	for line in soup.stripped_strings:
		result['stripped_strings'].append(line)

	result['tags_strings'] = []
	for tag in soup.find_all(True):
		result['tags_strings'].append(tag.name)

	if (debug):
		for idx, x in result.items():
			print(f"idx:{idx}\n {repr(x)[:100]}")
			print("\n\nXXXXXXXXXXXXXXXXXXXXXXXx\n\n")

	return result


def get_count_from_html(html_data, result: dict = {}, debug=0):
	# trigram_measures = nltk.collocations.TrigramAssocMeasures()
	# tokens = nltk.wordpunct_tokenize(html_data['text'])
	# finder = TrigramCollocationFinder.from_words(tokens)
	#     print(sorted(finder.nbest(trigram_measures.raw_freq, 5)))

	frequency = nltk.FreqDist(html_data['clean_text'].split())
	result['word_number'] = frequency.copy().items()
	max_freq_num = 1
	if (len(frequency.values()) > 0):
		max_freq_num = max(frequency.values())
	#     print(frequency.items()," == ",max_freq_num)
	for word in frequency.keys():
		frequency[word] = round(frequency[word] / max_freq_num, 4)
	result['word_freq'] = frequency.items()
	#     frequency.plot()

	# group patterns like at suffix tree find longest suffix list

	if (debug):
		for idx, x in result.items():
			print(f"idx:{idx}\n {str(x)[:700]}")
	return result


def get_nlp_count_from_text(text, result: dict = {}, debug=0):
	nlp = en_core_web_sm.load()
	# nlp = spicy.load('en_core_web_sm')
	doc = nlp(text)

	found_label = {}
	label_l = []
	for ent in doc.ents:
		#         print(ent.text, ent.start_char, ent.end_char, ent.label_)
		label_l.append(ent.label_)
		if (ent.label_ in found_label):
			found_label[ent.label_] += 1
		else:
			found_label[ent.label_] = 1
	result['spacy_label_l'] = label_l
	result['spacy_label'] = found_label

	if (debug):
		for idx, x in result.items():
			print(f"idx:{idx}\n {str(x)[:700]}")
	return result


def punctuation_count(text, result={}, debug=0):
	punctuation = {'punct_all': """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""",
				   'punct_bind': """()[]{}""",
				   'punct_compare': """<>=?""",
				   'punct_text': """"!',.:;`""",
				   'punct_other': """#$%&*+-/@\^_|~""",
				   'punct_hash': """#$@""",
				   'punct_math': """*+-%"""}
	for idx, punct in punctuation.items():
		# print(idx,"   ",punct)
		result[idx] = len([c for c in str(text) if c in punct])
	# can use also string.count
	if (debug):
		print(f"text:{text}\nresult:{result}")
	return result


def get_document_embeding(text, result={}, debug=0):
	from flair.data import Sentence
	from flair.embeddings import TransformerDocumentEmbeddings
	# embedding = TransformerDocumentEmbeddings('bert-base-cased')
	embedding = TransformerDocumentEmbeddings('distilbert-base-uncased')
	sentence = Sentence(text)

	# embed the sentence
	embedding.embed(sentence)
	embed = sentence.embedding.detach().tolist()
	result['doc_emb'] = embed
	if (debug):
		print(f"text:{text}\nresult:{result}")
	return result


def count_vector_most(docs, text_vec={}, debug=0):
	# https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.X7mOUC2l3OS
	if ('check_nltkdownload' not in locals()):
		nltk.download('stopwords')
		check_nltkdownload = True
	from nltk.corpus import stopwords
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer

	stop_words = set(stopwords.words('english'))
	# max_df = 0.85,
	cv = CountVectorizer(max_features=10000, stop_words=stop_words)
	word_count_vector = cv.fit_transform(docs)
	text_vec['word_count_vector'] = word_count_vector
	text_vec['cv_feature_names'] = cv.get_feature_names()
	text_vec['cv'] = cv

	# An extremely important point to note here is that the IDF should always be based on a large corpora and should be representative of texts you would be using to extract keywords.
	tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
	tfidf_transformer.fit(word_count_vector)
	text_vec['tfidf_transformer'] = tfidf_transformer
	if debug:
		# print idf values
		df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
		# sort ascending
		a = df_idf.sort_values(by=['idf_weights'])
		print(a)
	return text_vec


def tfidf_on_document_l(docs, text_vec={}, debug=0):
	#https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.X7onTC2l3UI
	from sklearn.feature_extraction.text import TfidfVectorizer

	# settings that you use for count vectorizer will go here
	tfidf_vectorizer = TfidfVectorizer(use_idf=True)

	# just send in all your docs here
	fitted_vectorizer = tfidf_vectorizer.fit(docs)
	text_vec['fitted_vectorizer'] = fitted_vectorizer
	text_vec['tfidf_vectorizer_vectors'] = fitted_vectorizer.transform(docs)

	if debug:
		# get the first vector out (for the first document)
		first_vector_tfidfvectorizer = text_vec['tfidf_vectorizer_vectors'][0]

		# place tf-idf values in a pandas data frame
		df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

		print(df)
		# sort ascending
		a = df.sort_values(by=['tfidf'], ascending=False)
		print(a)
	return text_vec

def sort_coo(coo_matrix):
	tuples = zip(coo_matrix.col, coo_matrix.data)
	return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def single_doc_tfidf(text, text_vec, result={}, debug=0):
	# generate tf-idf for the given document
	tf_idf_vector = text_vec['tfidf_transformer'].transform(text_vec['cv'].transform([text]))
	result['tf_idf_vector'] = tf_idf_vector

	score_vals = {}

	for idx, score in sort_coo(tf_idf_vector.tocoo()):
		score_vals[text_vec['cv_feature_names'][idx]] = round(score, 3)
	result['score_vals'] = score_vals

	if (debug):
		print(f"text:{text}\nresult:{result}")

	return result


def demo_func(debug=1, html_orj=""):
	if (html_orj == ""):
		html_orj = """<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">

	<title>Team | AeroMobil: Flying Car</title>
	<meta property="og:title" content="Team">
	<meta property="og:type" content="website">
		<meta name="robots" content="index, follow">
	<meta name="description" content="Welcome to our landing page and get ready for take-off! Aeromobil is a unique combination of car and airplane, a truly flying car.">
	<meta property="og:description" content="Welcome to our landing page and get ready for take-off! Aeromobil is a unique combination of car and airplane, a truly flying car.">
	<meta name="keywords" content="about,specification,people,evolution,media,presentedat,contact">

	<meta name="viewport" content="width=device-width">


		<link rel="canonical" href="https://www.aeromobil.com/team/">
		<meta property="og:url" content="https://www.aeromobil.com/team/">


	<link rel="stylesheet" href="/styles/app.min.css?v=4">	


	<script>
		(function(i,s,o,g,r,a,m) {
			i['GoogleAnalyticsObject']=r;i[r]=i[r]||function() {
				(i[r].q=i[r].q||[]).push(arguments)
			}, i[r].l=1*new Date();a=s.createElement(o),
			m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
		})
		(window,document,'script','//www.google-analytics.com/analytics.js','ga');
		ga('create', "UA-5422590-5", "aeromobil.com");
		ga('require', 'displayfeatures');
		ga('send', 'pageview');
	</script>


	<div id="fb-root"></div>
	<script>(function(d, s, id) {
		var js, fjs = d.getElementsByTagName(s)[0];
		if (d.getElementById(id)) return;
		js = d.createElement(s); js.id = id;
		js.src = "//connect.facebook.net/en_US/all.js#xfbml=1";
		fjs.parentNode.insertBefore(js, fjs);
		}(document, 'script', 'facebook-jssdk'));</script>

	<script>window.twttr = (function(d, s, id) {
		var js, fjs = d.getElementsByTagName(s)[0],
		t = window.twttr || {};
		if (d.getElementById(id)) return t;
		js = d.createElement(s);
		js.id = id;
		js.src = "https://platform.twitter.com/widgets.js";
		fjs.parentNode.insertBefore(js, fjs);

		t._e = [];
		t.ready = function(f) {
			t._e.push(f);
		};

		return t;
		}(document, "script", "twitter-wjs"));</script>
</head>

<body>

	<div class="w-page">
<header class="s-header">
	<nav class="navbar">
		<div class="container">
			<div class="navbar-header">
				<div class="e-logo-xs">
					<a href="/" title="Aeromobil">
						<img src="/assets/logo-ext.svg" alt="Aeromobil">
					</a>
				</div>

				<button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
					<span class="sr-only">Toggle navigation</span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span>

					<span>Menu</span>
				</button>
			</div>

			<div id="navbar" class="navbar-collapse collapse">
				<ul class="nav navbar-nav">
							<li><a href="/aeromobil-4_0-stol/">Flying Car<span class="e-menu-icon">...</span></a>
	<ul class="sub">
			<li>
				<a href="/aeromobil-4_0-stol/">AeroMobil 4.0 STOL</a>
			</li>
			<li>
				<a href="/aeromobil-5_0-vtol/">AeroMobil 5.0 VTOL</a>
			</li>
	</ul>
							</li>
							<li><a href="/evolution/">Evolution</a>
							</li>
							<li><a style="font-weight: bold" href="/team/">Team</a>
							</li>
							<li><a href="/partners/">Partners</a>
							</li>
							<li class="logo">
								<a href="/" title="Aeromobil"><img src="/assets/logo.png" alt="Aeromobil"></a>
							</li>
							<li><a href="/coverage/">Coverage</a>
							</li>
							<li><a href="/coverage/">Media<span class="e-menu-icon">...</span></a>
	<ul class="sub">
			<li>
				<a href="/official-news/">News</a>
			</li>
			<li>
				<a href="/events/">Events</a>
			</li>
	</ul>
							</li>
							<li><a href="/technical-services/">Technical Services</a>
							</li>
				</ul>
			</div>
		</div>
	</nav>
</header>



		<div class="content">
		<div class="c c-11">
<!-- Team -->
<style>.modal.carousel .modal-body .item header{margin-top:0;}</style>

<section class="s s-team  s-header-offset">
  <a name="team" id="team"></a>

  <div class="container">
	<header>
	  <div class="row">
		<div class="col-xs-12 col-sm-8 col-sm-offset-2 col-md-6 col-md-offset-3">
		  <img src="../../assets/logo-ext.svg" alt="#" class="img-responsive">
		  <p style="margin-top: 30px;">AeroMobil is an international team of more than forty experts from the automotive and aerospace sectors representing over eight countries across the world. Based in Central Europe they work together to design and manufacture the unique and innovative vehicle within all existing regulations for road and air transport.</p>
		</div>
		</div>
	</header>
<h3 class="s-title">Advisory board</h3>
<div class="b-team js-team">
	<div class="row">
		<div class="col-xs-12 col-sm-10 col-sm-offset-1">
			<div class="row">
				<div class="col-xs-12 col-sm-4 i">
					<a data-toggle="modal" data-target="#advisory-board" data-slide-to="62">
					<div class="image">
						<img src="/uploads/team/Patrick_Hessel.jpg" alt="Patrick Hessel" class="img-responsive">
					</div>
					<h4 class="title">Patrick Hessel</h4>
					<h5 class="subtitle">Chairman</h5>
					</a>
				</div>
				<div class="col-xs-12 col-sm-4 i">
					<a data-toggle="modal" data-target="#advisory-board" data-slide-to="6">
					<div class="image">
						<img src="/uploads/team/foto_01.jpg" alt="Juraj Vaculik" class="img-responsive">
					</div>
					<h4 class="title">Juraj Vaculik</h4>
					<h5 class="subtitle">Co-Founder, former CEO (2010-2019)</h5>
					</a>
				</div>
				<div class="col-xs-12 col-sm-4 i">
					<a data-toggle="modal" data-target="#advisory-board" data-slide-to="3">
					<div class="image">
						<img src="/uploads/team/foto_14.jpg" alt="Antony Sheriff" class="img-responsive">
					</div>
					<h4 class="title">Antony Sheriff</h4>
					<h5 class="subtitle">Advisor</h5>
					</a>
				</div>
				<div class="col-xs-12 col-sm-4 i">
					<a data-toggle="modal" data-target="#advisory-board" data-slide-to="4">
					<div class="image">
						<img src="/uploads/team/foto_35.jpg" alt="David Richards" class="img-responsive">
					</div>
					<h4 class="title">David Richards</h4>
					<h5 class="subtitle">Advisor</h5>
					</a>
				</div>
				<div class="col-xs-12 col-sm-4 i">
					<a data-toggle="modal" data-target="#advisory-board" data-slide-to="70">
					<div class="image">
						<img src="/uploads/team/arnon.jpg" alt="Meir Arnon" class="img-responsive">
					</div>
					<h4 class="title">Meir Arnon</h4>
					<h5 class="subtitle">Advisor</h5>
					</a>
				</div>
			</div>
		</div>
	</div>

	<div class="modal fade carousel slide js-team-modal" id="advisory-board">
		<div class="modal-dialog">
			<div class="modal-content">
				<div class="modal-header">
					<button type="button" class="close" data-dismiss="modal"><i class="icon icon-close"></i></button>
				</div>
				<div class="modal-body">
					<div class="carousel-inner">

						<div class="item" data-slide="62">
							<header>
								<div class="row">
									<div class="col-md-2">
										<img src="/uploads/team/Patrick_Hessel.jpg" alt="Patrick Hessel" class="img-responsive">
									</div>
									<div class="col-md-10">
										<h3 class="e-section-title">Patrick Hessel</h3>
										<h4 class="e-section-subtitle">Chairman</h4>
									</div>
								</div>
							</header>

							<div class="b-content">
								<p>An experienced and pragmatic entrepreneur in the high-technology mechanical engineering sector. Awarded EY Entrepreneur of the year 2015. Experience in setting-up divisions in engineering design, tooling production, prototype and series-production in high-growth, high value-add industries. A results orientated, strategic driver with operational infrastructure and planning skills.<br> <br> Patrick Hessel is a technology investor in AeroMobil and founder and CEO of c2i - an advanced composites engineering, tooling and production company. He developed a wide range of production technologies to produce light-weight advanced engineered products to high volume, lost cost and high-end quality standards. Built an international customer base for industries: automotive, aerospace, motorsport, satellite communication. c2i customer references include companies like BMW, Audi, Porsche, Bentley, Aston Martin, Jaguar, Mercedes Motorsport HWA, B/E Aerospace, Diehl, Alfa Romeo, Cobham. Before starting c2i, Patrick Hessel worked as a management consultant at McKinsey&amp;Co. in Germany.</p>
							</div>		
						</div>
						<div class="item" data-slide="6">
							<header>
								<div class="row">
									<div class="col-md-2">
										<img src="/uploads/team/foto_01.jpg" alt="Juraj Vaculik" class="img-responsive">
									</div>
									<div class="col-md-10">
										<h3 class="e-section-title">Juraj Vaculik</h3>
										<h4 class="e-section-subtitle">Co-Founder, former CEO (2010-2019)</h4>
									</div>
								</div>
							</header>

							<div class="b-content">
								<p>Juraj Vaculik is a co-founder and CEO of AeroMobil, an advanced engineering company that is commercialising a sophisticated flying car, combining real car and an aircraft in a single vehicle.</p><p>In 2010, Juraj co-founded AeroMobil and manages the company as its CEO. In 2013, together with the inventor and co-founder Stefan Klein, he unveiled the pre-prototype of AeroMobil 2.5 at the SAE Conference in Montreal. A year later, an experimental prototype of AeroMobil 3.0 was developed under his lead with the team of 12 people and presented at the Pioneers Festival in Vienna. Popular Science magazine presented AeroMobil with Invention of the Year 2015 award and UK edition od Wired magazine ranked the project among the top 10 rule breakers last year.</p><p>Juraj has over twenty years’ experience of working as a leader with very broad experiences from political revolution to international media and advertising industry. During Velvet Revolution in 1989 which ends a communist era in former Czechoslovakia, he was one of the key persons in Student Movement who have started the process of democratization and established the new post-revolution government in Slovakia.</p><p>In 1992, he started his career as a creative director for major global advertising agencies which started their operations in Czech and Slovak republics. In 1996 he founded MADE BY VACULIK - one of the leading independent advertising agencies in the CEE region, extending its reach to over 30 countries.</p><p>Juraj acts as an angel investor in numerous projects in Europe and the US.</p>
							</div>		
						</div>
						<div class="item" data-slide="3">
							<header>
								<div class="row">
									<div class="col-md-2">
										<img src="/uploads/team/foto_14.jpg" alt="Antony Sheriff" class="img-responsive">
									</div>
									<div class="col-md-10">
										<h3 class="e-section-title">Antony Sheriff</h3>
										<h4 class="e-section-subtitle">Advisor</h4>
									</div>
								</div>
							</header>

							<div class="b-content">
								<p style="box-sizing: border-box; margin: 0px 0px 10px; font-size: 16px; font-family: Sansation; ">Antony Sheriff is a renowned automotive industry professional. With an international career stretching over 30 years, Sheriff is recognized as one of the industry’s foremost product experts and has a deep understanding and knowledge of the auto industry at both the highest strategic levels and the detailed design and operating level.</p><p style="box-sizing: border-box; margin: 0px 0px 10px; font-size: 16px; font-family: Sansation; ">After briefly working as a car designer in Italy and England, he worked and as a product planner for Chrysler. Mr. Sheriff went on to receive his Masters in Management from M.I.T. where he was also a full time researcher for the International Motor Vehicle Program and contributed to the ground-breaking book, “The Machine That Changed the World”. He subsequently joined McKinsey &amp; Company where he advised a number of automotive and non-automotive companies throughout the US and Europe.</p><p style="box-sizing: border-box; margin: 0px 0px 10px; font-size: 16px; font-family: Sansation; ">In 1995, Sheriff joined Fiat Auto in Italy and was quickly promoted to Director of Product Development for all Fiat, Alfa Romeo and Lancia cars and commercial vehicles. Amongst the many cars he conceived, he was responsible for two “Car of the Year” winners, Alfa Romeo 147 and Fiat Panda. After being appointed Vice-President of the Fiat brand, Sheriff left to join McLaren Automotive in 2003 as Managing Director.</p><p style="box-sizing: border-box; margin: 0px 0px 10px; font-size: 16px; font-family: Sansation; ">In his decade at McLaren, Sheriff transformed the company from a contract manufacturer for Mercedes-Benz into a profitable independent sports car company with a full product range of highly innovative sports cars that have met broad critical acclaim. Initially, he launched the Mercedes SLR McLaren to market and then conceived and produced the SLR Stirling Moss model, the most expensive Mercedes ever sold.</p><p style="box-sizing: border-box; margin: 0px 0px 10px; font-size: 16px; font-family: Sansation; ">Sheriff transformed McLaren into a truly independent company and, under his leadership and vision, the company launched the MP4-12C, a revolutionary carbon fibre sports car, followed by the P1, a million dollar hybrid ultra-high performance sports car. A third car that Sheriff conceived, the Sports Series, is being launched later this year. The company also built a brand new production facility and a global distribution network. In its first full year in production, McLaren Automotive achieved a turnover of £267 million and near breakeven profitability. Sheriff resigned as Managing Director of McLaren in 2013 and is now advising and investing in a number of projects in the international automotive industry. Sheriff holds a B.A. and a B.S. from Swarthmore College as well as an M.S. from M.I.T.. He holds American and Italian nationality.</p>
							</div>		
						</div>
						<div class="item" data-slide="4">
							<header>
								<div class="row">
									<div class="col-md-2">
										<img src="/uploads/team/foto_35.jpg" alt="David Richards" class="img-responsive">
									</div>
									<div class="col-md-10">
										<h3 class="e-section-title">David Richards</h3>
										<h4 class="e-section-subtitle">Advisor</h4>
									</div>
								</div>
							</header>

							<div class="b-content">
								<p><span style="font-family: Sansation; font-size: 16px; ">David Richards, CBE founded Prodrive in 1984 and acted as a chairman of Aston Martin. He is one of the most respected figures in world motorsport. Mr. Richards has been instrumental in establishing Prodrive as a world-leading motorsport and technology business. He has also led two F1 teams – Benetton and BAR - as team principal and in the latter led the team to second place in the F1 Constructors’ World Championship in 2004. Today, he is actively involved in all aspects of the Prodrive group and works closely with the board to develop the business strategy, supporting the executive team in its new business activity.</span></p>
							</div>		
						</div>
						<div class="item" data-slide="70">
							<header>
								<div class="row">
									<div class="col-md-2">
										<img src="/uploads/team/arnon.jpg" alt="Meir Arnon" class="img-responsive">
									</div>
									<div class="col-md-10">
										<h3 class="e-section-title">Meir Arnon</h3>
										<h4 class="e-section-subtitle">Advisor</h4>
									</div>
								</div>
							</header>

							<div class="b-content">
								<p>Meir Arnon is an Israeli Entrepreneur, Investor and Turnaround Leader.</p><p>Arnon founded Focus Capital Group (FCG) <a href="http://www.focuscap.com/" data-htmlarea-external="1" rtekeep="1">www.focuscap.com</a>  in 1988 and currently serves as its Chairman and CEO. FCG has led many investments and transactions, some in partnership with Fortune 100 companies such as EDS, Frito Lay, P&amp;G, Kimberly Clark and others, and some by acquiring troubled publically traded industrial corporations in various industries such as Food, Communication Equipment, Heavy Machinery, Textiles, Automotive, Mobility and Energy Storage, turning them around and selling them to industrial and strategic buyers. Currently Meir Arnon serves as Chairman of Vulcan- Volta <a href="http://www.volta.ci.il/" data-htmlarea-external="1" rtekeep="1">www.volta.ci.il</a> , and a board member of several Automotive and Mobility startups boards.</p><p>Prior to founding FCG, Arnon Co- Founded and Chaired the New Dimension Software Co. that was later went public on NASDAQ and sold to BMC Software.</p><p>Meir has worked with many leading Automotive OEs and Tier 1s companies such as GM, FORD, Renault, Delphi, Lear and others.</p><p>Arnon is an active civic leader, member of the YPO where he Co- Founded of the Israeli, Monaco and the Russian Chapters and championed many events worldwide. He is a board member of Israel Innovation Institute and the Co-Founder and Chairman of EcoMotion <a href="http://www.ecomotion.org.il/" data-htmlarea-external="1" rtekeep="1">www.ecomotion.org.il</a>, a non for profit global interdisciplinary community of many thousands of innovators and entrepreneurs engaged in Smart mobility and Alternative Fuels.</p><p>Arnon holds a B.Sc. from the Technion, Israel Institute of Technology, an MBA from INSEAD, and is an HBS alumnus.</p>
							</div>		
						</div>
					</div>
					<a class="left carousel-control" href="#advisory-board" role="button" data-slide="prev">
						<span class="e-arrow e-arrow-left">
							<i class="icon icon-keyboard-arrow-left"></i>
							<span class="e-name"></span>
						</span>
					</a>
					<a class="right carousel-control" href="#advisory-board" role="button" data-slide="next">
						<span class="e-arrow e-arrow-right">
							<span class="e-name"></span>
							<i class="icon icon-keyboard-arrow-right"></i>
						</span>
					</a>
				</div>
			</div>
		</div>
	</div>
</div>
  </div>
</section>

		</div>
		</div>


		<footer class="s-footer">
			<div class="b-footer">
				<div class="container">
					<div class="row">
						<div class="col-xs-12 col-sm-4 col-lg-3 col-lg-offset-1 i">
							<h4 class="title">Get to know us</h4>
							<a href="/team/" title="Our Team">Our Team <i class="icon icon-caret-right"></i></a>
						</div>

						<div class="col-xs-12 col-sm-4 col-lg-3 i">
							<h4 class="title">Join the crew</h4>
							<a href="/careers/" title="Careers">Careers <i class="icon icon-caret-right"></i></a>
						</div>

						<div class="col-xs-12 col-sm-4 col-lg-4 i">
							<h4 class="title">Stay updated</h4>
							<iframe id="iforms-42494e64-cc77-47cb-b173-1655cc4e03e3" class="js-forms-iframe" src="https://forms.aeromobil.com/iframe/42494e64-cc77-47cb-b173-1655cc4e03e3" style="border:none; overflow:hidden; width:100%;"></iframe>
						</div>
					</div>
				</div>
			</div>

			<div class="b-footer-info">
				<div class="container">
					<div class="row">
						<div class="col-xs-12 col-sm-4 col-md-3 col-md-offset-1 i">
							2020 AeroMobil. All Rights Reserved.
						</div>

						<div class="col-xs-12 col-sm-4 col-md-4 i text-center">
							Prípojná 5, 821 06 Bratislava, Slovakia <span class="devider">|</span> <a href="tel:+421911088590" title="#">+421 911 088 590</a> <span class="devider">|</span> <a href="mailto:info@aeromobil.com" title="info@aeromobil.com">info@aeromobil.com</a>
						</div>

						<div class="col-xs-12 col-sm-4 col-md-3 i text-right socials">
							<a href="https://www.facebook.com/aeromobilcom" title="Facebook AeroMobil" target="_blank"><i class="icon icon-facebook"></i></a>
							<a href="https://www.linkedin.com/company/aeromobil" title="LinkedIn AeroMobil" target="_blank"><i class="icon icon-linkedin"></i></a>
							<a href="https://twitter.com/aeromobil" title="Twitter AeroMobil" target="_blank"><i class="icon icon-twitter"></i></a>
							<a href="https://www.youtube.com/c/aeromobil" title="YouTube AeroMobil" target="_blank"><i class="icon icon-youtube"></i></a>
							<a href="https://open.spotify.com/user/1214513701/playlist/1izBwPSDwc5yzBVCFtJ3eR" title="Aeromobil Spotify" target="_blank"><i class="icon icon-spotify"></i></a>
						</div>
					</div>
				</div>
			</div>
		</footer>

		<div class="b-cookies js-cookies-box">
			<div class="container">
				<div class="row">
					<div class="col-xs-12 col-sm-8 col-md-9 col-lg-10"><p>We use cookies on our site to enhance your user experience. You can disable cookies by changing your browser settings. <a href="/cookies/" title="Learn more about cookies">Learn more.</a></p></div> 
					<div class="col-xs-12 col-sm-4 col-md-3 col-lg-2 e-btn"><a href="#" title="#" class="btn btn-primary js-accept-cookies">Accept Cookies</a></div>
				</div>
			</div>
		</div>
	</div>

		<script src="/scripts/app.min.js?v=4"></script>

</body>
</html>
"""
	result = {}
	html_data = get_text_from_html(html_orj, debug=debug)
	html_count = get_count_from_html(html_data, debug=debug)
	# TODO add tokenize of word
	get_nlp_count_from_text(html_data['clean_text'], debug=debug)
	punctuation_count(html_data['text'], debug=debug)
	get_document_embeding(html_data['text'], debug=debug)
	text_vec = count_vector_most(html_data['text'])
	single_doc_tfidf(html_orj, text_vec=text_vec, debug=debug)
	return result

# demo_func()
