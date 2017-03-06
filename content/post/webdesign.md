+++
date = "2017-03-04T17:21:15Z"
title = "Web Design Wisdom"
description = "Wisdom gained from setting up MLNotebook"
topics = ["updates"]
tags = ["hugo","highlight.js","syntax","mathjax","social media","div","css"]

+++

I thought I'd give an overview of some of the wisdom I've gained from creating MLNotebook - my adventures in markdown... and the rest!

<h2 id="hugo"> Hugo </h2>

<h3 id="hugoSetup"> Setup </h3>

Hugo was relatively easy to setup, but I think some of the guides around could be a lot clearer particularly when it comes to hosting on Githib Pages. Firstly, make sure that you download Hugo [here](https://github.com/spf13/hugo/releases "Hugo Github") and extract it to `/usr/local/bin`. I renamed mine to "hugo". Check whether its properly installed with the command:

```bash
$ hugo -v
```

This will provide the version number. If not, add `/usr/local/bin` to your system path:

```bash
$ PATH=$PATH:/usr/local/bin
```

Creating a new site called "newsite" from scratch is the easy bit:

```bash
$ hugo new site ./newsite
```

<h3 id="themeAndOverrides">Theme and overrides </h3>

To get my theme to work, I simply cloned the repository (as shown [here](https://themes.gohugo.io/blackburn/ "Blackburn theme")) directly into ./newsite/themes/blackburn. Be sure to copy the `config.toml` file to `./newsite`. That's all there is to it!

```bash
$ mkdir themes
$ cd themes
$ git clone https://github.com/yoshiharuyamashita/blackburn.git
```

Customising this theme was really easy as it is mostly done in config.toml. What I wish I knew about Hugo straight off the bat is that the tree structure is important. So anything in the "themes" folder is a fall-back for anything that **isn't** present in the root folder of the site. That means if you have your own template for a post in `./newsite/layouts/single.html` it will be used instead of the themes one in `./newsite/themes/layouts/single.html`. Thus if you want to edit the layout, copy the theme's one into your sites layout folder and edit it from there.

The index page is the same deal, just copy it to your sites root and it will take precident over the default theme's one.

<h3 id="partials">Partials</h3>

The partials bit can be a little confusing if you're not too familiar with how the site is put together. Effectively, the page you're loooking at right now is made up of lots of different parts (partials) that have been edited separately, put through a parser, turned into HTML and pasted together into a single HTML page. The head and footer don't have much in them but are important for adding calls to Javascripts as they are stitched into each and every page on the website. Don't confuse the head.html and header.html files, the latter is the actual title/banner at the top of the homepage (it is another partial that is stitched into index.html.

<h3 id="socialMediaButtons">Social Media Buttons</h3>

I spend a while trying to figure out how to get my social media buttons to actually take the url of the page they were on and share that exact post. I tried a hosted service which gave me a script that pulled down the buttons from them and allowed me to edit them via their interface, but it wasn't content-specific. To dynamically get the url and get some nice-looking icons, I actually used the site [Simple Sharing Buttons](https://simplesharingbuttons.com/ "Simple Sharing Buttons"), chose the sites I wanted and theyprovided the icons along with the HTML. In comparisson to other sites and methods, this seems to work the best (except for the reddit one really).

```html
<ul class="share-buttons">
  <li><a href="https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fmlnotebook.github.io&t=" title="Share on Facebook" target="_blank" onclick="window.open('https://www.facebook.com/sharer/sharer.php?u=' + encodeURIComponent(document.URL) + '&t=' + encodeURIComponent(document.URL),'','width=500,height=300'); return false;"><img alt="Share on facebook" src="/img/facebook.png"></a></li>
  <li><a href="https://twitter.com/intent/tweet?source=https%3A%2F%2Fmlnotebook.github.io&text=:%20https%3A%2F%2Fmlnotebook.github.io&via=mlnotebook" target="_blank" title="Tweet" onclick="window.open('https://twitter.com/intent/tweet?text=' + encodeURIComponent(document.title) + ':%20'  + encodeURIComponent(document.URL),'','width=500,height=300'); return false;"><img alt="Tweet" src="/img/twitter.png"></a></li>
  <li><a href="http://www.reddit.com/submit?url=https%3A%2F%2Fmlnotebook.github.io&title=" target="_blank" title="Submit to Reddit" onclick="window.open('http://www.reddit.com/submit?url=' + encodeURIComponent(document.URL) + '&title=' +  encodeURIComponent(document.title),'','width=500,height=300'); return false;"><img alt="Submit to Reddit" src="/img/reddit.png"></a></li>
  <li><a href="http://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fmlnotebook.github.io&title=&summary=&source=https%3A%2F%2Fmlnotebook.github.io" target="_blank" title="Share on LinkedIn" onclick="window.open('http://www.linkedin.com/shareArticle?mini=true&url=' + encodeURIComponent(document.URL) + '&title=' +  encodeURIComponent(document.title),'','width=500,height=300'); return false;"><img alt="Share on LinkedIn" src="/img/linkedin.png"></a></li>
</ul>
```

<h3 id="githubPages"> Hosting on Peronal Github Pages </h3>

Again, some of the tutorials out there aren't great at properly explaining how to get your pages hosted on your **personal** Github pages, rather than project ones (i.e. `https://<your username>.github.io`) I'll try to give you another version here.

Firstly, login to Github and create the repository `<your username>.github.io`. This is important as the master branch will be used to locate your website at exactly `https://<your username>.github.io`. Initialise it with the `README.md`. Create a new branch called `hugo` and initialise this with the `README.md` too.

In your `./newsite` directory you'll need to build the site, initialise the git respository and add the remote:

```bash
$ hugo
$
$ git init
$ git remote add origin git@github.com:<username>/<username>.github.io.git
```
If you're having trouble adding the remote because of _permissions_ it could be that you're using a different Git account for your website than normal. Have a look at the `git config` options to change the username/password. If that fails, it could be that you need to sort an `ssh` key - instructions for that are on your account settings page.

From here, I managed to find and adapt two scripts from [here](https://hjdskes.github.io/blog/deploying-hugo-on-personal-gh-pages/ "hjdskes"). The first is `setup.sh` ([download](/docs/setup.sh "setup.sh")) and only needs to be executed once. It does the following:


* Deletes the master branch (perfectly safe)
* Creates a new orphaned master branch
* Takes the `README.md` from `hugo` and makes an initial commit to `master`
* Changes back to `hugo`
* Removes the existing `./public` folder
* Sets the `master` branch as a subtree for the `./public` folder
* Pulls the commited `master` back into `./public` to stop merge conflicts.

<div class="warn">Make sure that you edit the `USERNAME` field in `setup.sh` before executing.</div>

After that, whenever you want to upload your site, just run the second script `deploy.sh` which I've altered slightly ([download](/docs/deploy.sh "deploy.sh")) with an optional argument which will be your commit message: missing out the argument submits a default message.

`deploy.sh` commits and pushes all of your changes to the `hugo` source branch before putting the `./public` folder on `master`.

<div class="warn">Make sure that you edit the `USERNAME` field in `deploy.sh` before executing</div>

And that's it! If the website doesn't load when you go to `https://<your username>.github.io` you may need to hit `settings` in your repo (top right of the menu bar), scroll down to "Github Pages" and select `master` as your source.

<h2 id="htmlCss">HTML / CSS</h2>

<h3 id="contactForm">Contact Form</h3>

The first part of the site I altered was the contant page. I added a contact form which largely involves `html` formatted with `css`. The magic that makes it work comes from the free service called [Formspree](https://formspree.io/ "Formspree"). Essentially, the submit button sends the information to formspree and they forward it on to me directly. It uses a hidden field to give the forwarded emails the same subject, this makes for easy filtering. It also provides a free "I'm not a robot" page after clicking submit.

```html
<div id="contactform" class="center">
<form action="https://formspree.io/your@email.com method="POST" name="sentMessage" id="contactForm" novalidate>
	<input type="text" name="name" placeholder="Name" id="name" required data-validation-required-message="Please enter your name."><br>
	<input type="email" name="_replyto" placeholder="Email Address" id="email" required data-validation-required-message="Please enter your email address." ><br>

	<input type="hidden"  name="_subject" value="Message from MLNotebook">
	<input type="text" name="_gotcha" style="display:none" />
	<textarea rows="10" name="message" class="form-control" placeholder="Message" id="message" required data-validation-required-message="Please enter a message."></textarea><br>
	<input type="submit" value="Send">
</form>
</div>
```

The formatting was a pain as I'd never used the box-size argument before - this is what I found made the boxes all the same size and have the same alignment. I added for all browsers too.

```css

input[type=text], input[type=email], textarea {
	display: inline-block;
  	border: 1px solid transparent;
  	border-top: none;
  	border-bottom: 1px solid #DDD;
  	box-shadow: inset 0 1px 2px rgba(0,0,0,.39), 0 -1px 1px #FFF, 0 1px 0 #FFF;
	border-radius: 4px;
	margin: 2px 2px 2px 2px;
	resize:none;
	float: left;
	width: 100%;
}

textarea, input {
    -webkit-box-sizing: border-box;
    -moz-box-sizing: border-box;
    box-sizing: border-box;
}

input[type=submit] {
	width: 100%;
}

.center {
	margin: auto;
}

input {
	height:50px;
}

textarea {
	height: 200px;
	padding-left: 0px;
}

input, textarea::-webkit-input-placeholder {
   padding-left: 10px;
}
input, textarea::-moz-placeholder {
   padding-left: 10px;
}
input, textarea:-ms-input-placeholder {
   padding-left: 10px;
}
input, textarea:-moz-placeholder {
   padding-left: 10px;
}
  
  ```

<h3	id="resizing">Resizing for Small Screens</h3>

One of my final hurdles in getting the site setup was making the homepage a little more friendly that just showing the recent posts. So I decided to add my [twitter](https://twitter.com/mlnotebook "@MLNotebook") feed to the side. Twitter has an easy code to embed this, and I just put it into its own partial in `layouts/partials/twitterfeed.html`.

My problem here though was that when I viewed my site on my phone, or resized the web-browser on the computer, the content would shrink and be almost unreadable - I wanted the feed to move below the text if the screen was below a certain size. So I created the usual `div` containers within my `index.html` file and added the shortcode to include my `twitterfeed.html` in the right-hand side.

```html
<div id="container" class="center">
	<div id="left_content" class="center">
		<div class="content">
		  {{ range ( .Paginate (where .Data.Pages "Type" "post")).Pages }}
		    {{ .Render "summary"}}
		  {{ end }}

		  {{ partial "pagination.html" . }}
		</div>
	</div>
	<div id="right_content" class="center">
		<center>{{ partial "twitterfeed.html" . }}</center>
	</div>
</div>
```

I then used `css` to give the `div` containers their own properties for different screen sizes:

```css
#container {
	position: relative;
	width:auto;

}

#right_content {
	float:left;
	overflow:hidden;
	display:block;
	padding-right:1%;

}

#left_content {
	float:left;
	width:80%;
	display:block;
	margin:auto;
	min-width=600px;

}

pre > code {
	font-size:11pt;
}

@media screen and (max-width: 1000px) {

#left_content {
	width: 100%;
	}
	
	.content {
	max-width:100%;
	}



#right_content {
	width:100%;
}

pre > code {
	font-size:8pt;
}

}

```

Note that this allows the size of the font in the code-snippets to shrink when the screensize is small - I find that it reads more easily.

<h2 id="syntaxHighlighting">Syntax highlighting</h2>

So actually getting code into the website was trickier than I thought. The in-built markdown codeblocks seem to work just fine by adding code between backticks: `` `<code here>` ``. Markdown doesn't do syntax highlightsing right out of the box though. So I'm using `highlight.js`. My theme does come with a highlight shortcode option, but I found that I couldn't customise it how I wanted - particuarly, the font size was just too big. I tried everything, even adding extra `<pre> </pre>` tags around it and using `css` to format them. In the end, I found that using `highlight.js` was much simpler - I just loaded the script straight off their server and voila! The link just needed editing to select the theme I wanted, but I opted for the standard `monokai` anyway. I placed this in my site's `head` partial.

```html
<link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/styles/monokai.min.css">
<script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/highlight.min.js"></script>
<script>hljs.initHighlightingOnLoad();</script>
```

<h2 id="mathsRendering">Maths Rendering</h2>

Being a site on machine learning, I'm going to need to be able to include some mathematics sometimes. I'm very familiar with $\rm\LaTeX$ and I've written-up a lot of formulae already, so I looked into getting $\rm\LaTeX$ formatting into markdown/Hugo. A few math rendering engines are around, but not all are simple to implement. The best option I found was [MathJax](https://www.mathjax.org/ "MathJax") which literally required me to add these few lines to my `head` partial.

```javascript
<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']],
    processEscapes: true,
    processEnvironments: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    TeX: { equationNumbers: { autoNumber: "AMS" },
         extensions: ["AMSmath.js", "AMSsymbols.js"] }
  }
});
</script>
```

From there, it allows me to put inline math into my websites such as $ c = \sqrt{a^{2} + b^{2}} $ by enclosing them in the normal \$ symbols like so: `\$ some math \$`. MathJax also provides display-style input with enclosing `<div>\$\$ code \$\$</div>` e.g.: 

<div>$$ c = \sqrt{a^{2} + b^{2}}  $$</div>

The formatting is done by some css

```css
code.has-jax {
	font: inherit;
	font-size: 100%;
	background: inherit;
	border: inherit;
	color: #515151;
}
```


</td>

