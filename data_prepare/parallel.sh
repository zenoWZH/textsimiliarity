for year in {2003..2014}
do
    (
        python retrieve_news_jpenzh.py --retrieve=True --year=$year --preprocess=True --textout=True
    )& 
    wait -4
done
