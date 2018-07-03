import numpy as np
import os
import re
import h5py
import math
import json
from collections import Counter

def create_vocabulary_word2vec(file, capl=None, v2i={'': 0, 'UNK':1, 'BOS':2, 'EOS':3}, word_threshold=2, sen_length=5):
	'''
	v2i = {'': 0, 'UNK':1}  # vocabulary to index
	limit_sen: the number sentence for training per video
	'''
	json_file = file+'/videodatainfo_2017.json'
	train_info = json.load(open(json_file,'r'))
	videos = train_info['videos']
	sentences = train_info['sentences']
	train_video = [v['video_id'] for v in videos if v['id']<=6512]
	val_video = [v['video_id'] for v in videos if v['id']>=6513 and v['id']<=7009]
	test_video = [v['video_id'] for v in videos if v['id']>=7010]

	train_data = []
	val_data = []
	test_data = []

	

	print('preprocess sentence...')
	for idx, sentence in enumerate(sentences):
		video_id = sentence['video_id']
		caption = sentence['caption'].strip().split(' ')
		# print caption
		if(video_id in train_video):
			if len(caption)<capl and len(caption)>=sen_length:
				train_data.append({video_id:caption})
				
			
		elif(video_id in val_video):
			if len(caption)<capl and len(caption)>=sen_length:
				val_data.append({video_id:caption})
				
			
		# elif(video_id in test_video):
		# 	# test_data.append({video_id:caption})
		# 	test_data.append({video_id:['']})
	def generate_test_data():
		captions = []
		
		for idx in range(7010,10000):
			cap = {}
			cap['video'+str(idx)] = ['']
			captions.append(cap)
		return captions

	test_data = generate_test_data()
	print('build vocabulary...')
	all_word = []
	for data in train_data:
		for k,v in data.items():
			all_word.extend(v)
	for data in val_data:
		for k,v in data.items():
			all_word.extend(v)

	vocab = Counter(all_word)
	vocab = [k for k in vocab.keys() if vocab[k] >= word_threshold]

	# create vocabulary index
	for w in vocab:
		if w not in v2i.keys():
			v2i[w] = len(v2i)

	# new training set and validation set
	
	# if limit_sen is None:

	print('size of vocabulary: %d '%(len(v2i)))
	print('size of train, val, test: %d, %d, %d' %(len(train_data),len(val_data),len(test_data)))
	return v2i, train_data, val_data, test_data	
	

def getCategoriesInfo(file):
	'''
	v2i = {'': 0, 'UNK':1}  # vocabulary to index
	limit_sen: the number sentence for training per video
	'''
	json_file = file+'/videodatainfo_2017.json'
	train_info = json.load(open(json_file,'r'))
	videos = train_info['videos']
	cate_info = {}
	for idx,video in enumerate(videos):
		cate_info[video['video_id']]=video['category']

	return cate_info

def getBatchVideoCategoriesInfo(batch_caption, cate_info, feature_shape):
	batch_size = len(batch_caption)
	input_categories = np.zeros((batch_size,1),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			input_categories[idx,0] = cate_info[k]
	return input_categories

def getBatchVideoAudioInfo(batch_caption, audio_info):
	batch_size = len(batch_caption)
	input_audio = np.zeros((batch_size,34,2),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			vid = int(k[5:])
			input_audio[idx,:,:] = audio_info[vid]
	return input_audio

def generate_vocab(train_data, v2i={'': 0, 'UNK':1, 'BOS':2, 'EOS':3}):


	for caption_info in train_data:
		for k,v in caption_info.items():
			for w in v:
				if not v2i.has_key(w):
					v2i[w] = len(v2i)


	print('vocab size %d' %(len(v2i)))
	return v2i
	


def getBatchVideoFeature(batch_caption, hf, feature_shape):
	batch_size = len(batch_caption)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			feature = hf[k.split('/')[1]]
			# print(feature.shape)
			input_video[idx] = np.reshape(feature,feature_shape)
	return input_video

def getBatchC3DVideoFeature(batch_caption, hf, feature_shape):
	batch_size = len(batch_caption)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			vid = int(k[5:])
			feature = hf[vid]
			input_video[idx] = np.reshape(feature[0:40,:],feature_shape)
	return input_video

def getBatchStepVideoFeature(batch_caption, hf, feature_shape):
	batch_size = len(batch_caption)
	feature_shape = (40,1024)
	step = np.random.randint(1,5)
	# print(step)
	input_video = np.zeros((batch_size,)+tuple((10,1024)),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			vid = int(k[5:])
			feature = hf[vid]
			input_video[idx] = np.reshape(feature,feature_shape)[0::step][0:10]
	return input_video



def getBatchTrainCaption(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)

	labels = np.zeros((batch_size,capl,len(v2i)),dtype='int32')

	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']

	for idx, caption in enumerate(batch_caption):
		for vid, sen in caption.items():

			for k, w in enumerate(sen):
				
				if w in v2i.keys():
					labels[idx][k][v2i[w]] = 1
					input_captions[idx][k+1] = v2i[w]
				else:
					labels[idx][k][v2i['UNK']] = 1
					input_captions[idx][k+1] = v2i['UNK']
			labels[idx][len(sen)][v2i['EOS']] = 1
			if len(sen)+1<capl:
				input_captions[idx][len(sen)+1] = v2i['EOS']
	return input_captions, labels



def getBatchTestCaption(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)
	labels = np.zeros((batch_size,capl,len(v2i)),dtype='int32')
	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']


	return input_captions, labels

def getBatchTrainCaptionWithSparseLabel(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)

	labels = np.zeros((batch_size,capl),dtype='int32')

	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']

	for idx, caption in enumerate(batch_caption):
		for vid, sen in caption.items():

			for k, w in enumerate(sen):
				
				if w in v2i.keys():
					labels[idx][k]=v2i[w] 
					input_captions[idx][k+1] = v2i[w]
				else:
					labels[idx][k]= v2i['UNK']
					input_captions[idx][k+1] = v2i['UNK']
			labels[idx][len(sen)]= v2i['EOS']
			if len(sen)+1<capl:
				input_captions[idx][len(sen)+1] = v2i['EOS']
	return input_captions, labels




def getBatchTestCaptionWithSparseLabel(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)
	labels = np.zeros((batch_size,capl),dtype='int32')
	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']


	return input_captions, labels
def convertCaptionI2V(batch_caption, generated_captions,i2v):
	captions = []
	for idx, sen in enumerate(generated_captions):
		caption = ''
		for word in sen:
			if i2v[word]=='EOS' or i2v[word]=='':
				break
			caption+=i2v[word]+' '
		captions.append(caption)
	return captions


if __name__=='__main__':
	main()