
.PHONY: clean

all: executed-posts plugins/graphviz/graphviz.plugin
	nikola build

POSTS=$(shell find posts-sources -not -path '*/\.*' -type f -name '*\.ipynb')
executed-posts: $(patsubst posts-sources/%.ipynb,posts/%.ipynb,$(POSTS))

posts/%.ipynb: posts-sources/%.ipynb
	jupyter nbconvert \
	    --to ipynb \
	    --execute \
	    --ExecutePreprocessor.timeout=24000 \
	    --output-dir=./posts \
	    --output=`basename $@` $<

plugins/graphviz/:
	nikola plugin --install graphviz

clean:
	rm -rf posts
	nikola clean
