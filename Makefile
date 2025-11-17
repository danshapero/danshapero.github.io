
.PHONY: clean

all: executed-posts plugins/graphviz/graphviz.plugin
	nikola build

POSTS=$(shell find posts-sources -not -path '*/\.*' -type f -name '*\.py')
executed-posts: $(patsubst posts-sources/%.py,posts/%.ipynb,$(POSTS))

posts/%.ipynb: posts-sources/%.py
	jupytext --to ipynb --execute --output $@ $<

plugins/graphviz/graphviz.plugin:
	nikola plugin --install graphviz

clean:
	rm -rf posts
	nikola clean
