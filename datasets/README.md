# textgenrnn Datasets

Here are a few datasets for experimenting with textgenrnn. All datasets are obtained by inputing the given query into [BigQuery](https://cloud.google.com/bigquery/)

## Hacker News

Top 2000 Hacker News submissions by score. To reproduce:

```sql
#standardSQL
SELECT title
FROM `bigquery-public-data.hacker_news.full`
WHERE type = 'story'
ORDER BY score DESC
LIMIT 2000
```

Save to Google Sheets, and download as a .tsv (*not* a .csv, as csvs enclose sentences with quotes which contains commas!)

Replace `2000` with a larger number if necessary (up to `10000`)

## Reddit Subreddit Data

Top 1000 submissions by score for each of the included subreddits in the query, from January 2017 to June 2017. To reproduce in BigQuery:

```sql
#standardSQL 
SELECT title FROM (
SELECT title,
  ROW_NUMBER() OVER (PARTITION BY subreddit ORDER BY score DESC) as score_rank
  FROM `fh-bigquery.reddit_posts.*`
  WHERE (_TABLE_SUFFIX BETWEEN '2017_01' AND '2017_06')
  AND LOWER(subreddit) IN ("legaladvice", "relationship_advice")
  )
WHERE score_rank <= 1000
```

Save to Google Sheets, and download as a .tsv (*not* a .csv, as csvs enclose sentences with quotes which contains commas!)

Change `1000` and the included subreddits as appropriate (make sure the total does not exceed 10000!)