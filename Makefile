
.PHONY: clean

all: executed-posts theme

theme: themes/maupassant
	nikola theme -i maupassant

POSTS=$(shell find posts-sources -not -path '*/\.*' -type f -name '*\.ipynb')
executed-posts: $(patsubst posts-sources/%.ipynb,posts/%.ipynb,$(POSTS))

posts/%.ipynb: posts-sources/%.ipynb
	jupyter nbconvert \
	    --to ipynb \
	    --execute \
	    --ExecutePreprocessor.timeout=24000 \
	    --output-dir=./posts \
	    --output=`basename $@` $<

clean:
	rm -rf posts
	nikola clean
