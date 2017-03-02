// JavaScript Document
'use strict';

	var currentURL = document.URL;
	var currentTitle = document.title;

function socialShare() {
	var fbShare = document.getElementById("fbShare");
	var gplusShare = document.getElementById("gplusShare");
	var twitterShare = document.getElementById("twitterShare");
	var linkedinShare = document.getElementById("linkedinShare");
	
	fbShare.onclick = function() {
		window.open("https://www.facebook.com/sharer.php?u="+currentURL,"","height=368,width=600,left=100,top=100,menubar=0");
		return false;
	}
	gplusShare.onclick = function() {
		window.open("https://plus.google.com/share?url="+currentURL,"","height=550,width=525,left=100,top=100,menubar=0");
		return false;
	}	
	twitterShare.onclick = function() {
		window.open("https://twitter.com/share?url="+currentURL+"&text="+currentTitle,"","height=260,width=500,left=100,top=100,menubar=0");
		return false;
	}
	linkedinShare.onclick = function() {
		window.open("https://www.linkedin.com/cws/share?url="+currentURL,"","height=260,width=500,left=100,top=100,menubar=0");
		return false;
	}	
	
	fbShare.setAttribute("href","http://www.facebook.com/sharer.php?u="+currentURL);
	gplusShare.setAttribute("href","https://plus.google.com/share?url="+currentURL);
	twitterShare.setAttribute("href","https://twitter.com/share?url="+currentURL+"&text="+currentTitle);
	linkedinShare.setAttribute("href","https://www.linkedin.com/cws/share?url="+currentURL);
}


window.onload = function() {
	socialShare();
}