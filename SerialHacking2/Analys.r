my.data <- read.table('testcontracts.json')
length(my.data)
t.test <- my.data
my.corpus <- generateCorpus(my.data)

generateWordcloud <- function(corpus, min.freq = 3, ...) {
    require(wordcloud)
	doc.m <- TermDocumentMatrix(corpus,
        control = list(minWordLength = 1))
	dm <- as.matrix(doc.m)
	
	v <- sort(rowSums(dm), decreasing = TRUE)
    d <- data.frame(word = names(v), freq = v)
 
    require(RColorBrewer)
    pal <- brewer.pal(8, "Accent")
	
	wc <- wordcloud(d$word, d$freq, min.freq = min.freq, colors = pal)
    wc
}
my.corpus <- generateCorpus(tweets, с(my.stopwords, stops))
write.table(x, "clipboard", sep ="\t", col.names = NA)
aov(formula) 
save(test.csv)
