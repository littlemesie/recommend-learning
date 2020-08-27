from gensim import corpora, models, similarities
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from utils import rating_util

stop_list = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which',
             'one']


def title_document(title):
    document = []
    for t in title:
        document.append(t.get('title'))
    return document


def title_corpus(title):
    """"""
    frequency = defaultdict(int)
    texts = [[word for word in t['title'].lower().replace('\n', '').split() if word not in stop_list and not word.isdigit()] for t in title]
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    texts = [text for text in texts if text]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary


def tf_idf_model(document):
    """tf-idf模型"""
    tfidf_model = TfidfVectorizer()
    sparse_result = tfidf_model.fit_transform(document)
    vocabulary = tfidf_model.vocabulary_
    print(vocabulary)
    print(sparse_result)


def lda_model(title):
    """LDA模型"""
    corpus, dictionary = title_corpus(title)
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

    a = lda.show_topic(0, 10)
    print(a)

def lsi_model(title):
    """LSI模型"""
    corpus, dictionary = title_corpus(title)
    print(dictionary)
    print(corpus)
    lsi = gensim.models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=20)
    a = lsi.show_topic(0, 10)
    print(a)



if __name__ == '__main__':

    title = [{'title': 'One Flew Over the Cuckoo s Nest  1975', 'genres': 'Drama'}, {'title': 'James and the Giant Peach  1996', 'genres': "Animation Children's Musical"}, {'title': 'Flew Over Cuckoo s Nest', 'genres': 'Musical Romance'}, {'title': 'Erin Brockovich  2000', 'genres': 'Drama'}, {'title': 'Bug s Life  A  1998', 'genres': "Animation Children's Comedy"}, {'title': 'Princess Bride  The  1987', 'genres': 'Action Adventure Comedy Romance'}, {'title': 'Ben-Hur  1959', 'genres': 'Action Adventure Drama'}, {'title': 'Christmas Story  A  1983', 'genres': 'Comedy Drama'}, {'title': 'Snow White and the Seven Dwarfs  1937', 'genres': "Animation Children's Musical"}, {'title': 'Wizard of Oz  The  1939', 'genres': "Adventure Children's Drama Musical"}, {'title': 'Beauty and the Beast  1991', 'genres': "Animation Children's Musical"}, {'title': 'Gigi  1958', 'genres': 'Musical'}, {'title': 'Miracle on 34th Street  1947', 'genres': 'Drama'}, {'title': 'Ferris Bueller s Day Off  1986', 'genres': 'Comedy'}, {'title': 'Sound of Music  The  1965', 'genres': 'Musical'}, {'title': 'Airplane   1980', 'genres': 'Comedy'}, {'title': 'Tarzan  1999', 'genres': "Animation Children's"}, {'title': 'Bambi  1942', 'genres': "Animation Children's"}, {'title': 'Awakenings  1990', 'genres': 'Drama'}, {'title': 'Big  1988', 'genres': 'Comedy Fantasy'}, {'title': 'Pleasantville  1998', 'genres': 'Comedy'}, {'title': 'Wallace & Gromit  The Best of Aardman Animation  1996', 'genres': 'Animation'}, {'title': 'Back to the Future  1985', 'genres': 'Comedy Sci-Fi'}, {'title': 'Schindler s List  1993', 'genres': 'Drama War'}, {'title': 'Meet Joe Black  1998', 'genres': 'Romance'}, {'title': 'Pocahontas  1995', 'genres': "Animation Children's Musical Romance"}, {'title': 'E.T. the Extra-Terrestrial  1982', 'genres': "Children's Drama Fantasy Sci-Fi"}, {'title': 'Titanic  1997', 'genres': 'Drama Romance'}, {'title': 'Ponette  1996', 'genres': 'Drama'}, {'title': 'Close Shave  A  1995', 'genres': 'Animation Comedy Thriller'}, {'title': 'Antz  1998', 'genres': "Animation Children's"}, {'title': 'Girl  Interrupted  1999', 'genres': 'Drama'}, {'title': 'Hercules  1997', 'genres': "Adventure Animation Children's Comedy Musical"}, {'title': 'Aladdin  1992', 'genres': "Animation Children's Comedy Musical"}, {'title': 'Mulan  1998', 'genres': "Animation Children's"}, {'title': 'Hunchback of Notre Dame  The  1996', 'genres': "Animation Children's Musical"}, {'title': 'Last Days of Disco  The  1998', 'genres': 'Drama'}, {'title': 'Cinderella  1950', 'genres': "Animation Children's Musical"}, {'title': 'Sixth Sense  The  1999', 'genres': 'Thriller'}, {'title': 'Apollo 13  1995', 'genres': 'Drama'}, {'title': 'Toy Story  1995', 'genres': "Animation Children's Comedy"}, {'title': 'Rain Man  1988', 'genres': 'Drama'}, {'title': 'Driving Miss Daisy  1989', 'genres': 'Drama'}, {'title': 'Run Lola Run  Lola rennt   1998', 'genres': 'Action Crime Romance'}, {'title': 'Star Wars  Episode IV - A New Hope  1977', 'genres': 'Action Adventure Fantasy Sci-Fi'}, {'title': 'Mary Poppins  1964', 'genres': "Children's Comedy Musical"}, {'title': 'Dumbo  1941', 'genres': "Animation Children's Musical"}, {'title': 'To Kill a Mockingbird  1962', 'genres': 'Drama'}, {'title': 'Saving Private Ryan  1998', 'genres': 'Action Drama War'}, {'title': 'Secret Garden  The  1993', 'genres': "Children's Drama"}, {'title': 'Toy Story 2  1999', 'genres': "Animation Children's Comedy"}, {'title': 'Fargo  1996', 'genres': 'Crime Drama Thriller'}, {'title': 'Dead Poets Society  1989', 'genres': 'Drama'}]
    print(len(title))
    # document = title_document(title)
    # tf_idf_model(document)
    # lda_model(title)
    lsi_model(title)
