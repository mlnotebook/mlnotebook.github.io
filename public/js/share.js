// JavaScript Document
'use strict';

	var currentURL = document.URL;
	var currentTitle = document.title;

function socialShare() {
	var fbShare = document.getElementById("fbShare");
	var twitterShare = document.getElementById("twitterShare");
	var linkedinShare = document.getElementById("linkedinShare");

	twitterShare.onclick = function() {
		window.open("https://twitter.com/share?url="+currentURL+"&text="+currentTitle,"","height=260,width=500,left=100,top=100,menubar=0");
	}
	
	fbShare.onclick = function() {
		window.open("https://www.facebook.com/sharer.php?u="+currentURL,"","height=368,width=600,left=100,top=100,menubar=0");
	}
	linkedinShare.onclick = function() {
		window.open("https://www.linkedin.com/cws/share?url="+currentURL,"","height=260,width=500,left=100,top=100,menubar=0");
	}	
	fbShare.setAttribute("href","http://www.facebook.com/sharer.php?u="+currentURL);
	twitterShare.setAttribute("href","https://twitter.com/share?url="+currentURL+"&text="+currentTitle);
	linkedinShare.setAttribute("href","https://www.linkedin.com/cws/share?url="+currentURL);
}


window.onload = function() {
	socialShare();
}