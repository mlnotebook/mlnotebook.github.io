+++
date = "2017-03-01T12:46:58Z"
title = "Contact"
description = ""
topics = []
tags = []
permalink = "/contact/"
+++

Please leave any comments via Twitter or Github (links are on the left) or for anything else, use the form below.

<!-- Contact Section -->
<div id="contactform" class="center">
<form action="https://formspree.io/r.robinson16@imperial.ac.uk" method="POST" name="sentMessage" id="contactForm" novalidate>
	<input type="text" name="name" placeholder="Name" id="name" required data-validation-required-message="Please enter your name."><br>
	<input type="email" name="_replyto" placeholder="Email Address" id="email" required data-validation-required-message="Please enter your email address." ><br>

	<input type="hidden"  name="_subject" value="Message from MLNotebook">
	<input type="text" name="_gotcha" style="display:none" />
	<textarea rows="10" name="message" class="form-control" placeholder="Message" id="message" required data-validation-required-message="Please enter a message."></textarea><br>
	<input type="submit" value="Send">
</form>
</div>