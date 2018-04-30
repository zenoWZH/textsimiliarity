for year in {2003..2013}
do
    (
        python retreive_news.py --year=$year --preprocess=True --textout=True
    )&
done
