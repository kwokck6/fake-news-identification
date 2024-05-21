#!/bin/bash
#SBATCH --job-name=bert_fake_news
#SBATCH --mail-user=1155141911@link.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept8/fyp22/ktl2201/seem2460/bert_fake_news.txt
#SBATCH --gres=gpu:1
time python3 bert_fake_news.py
