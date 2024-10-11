/*
 * LeaveNotice - plug in to notify users of leaving your site
 * Examples and documentation at: http://rewdy.com/tools/leavenotice-jquery-plugin
 * Version: 1.1.3 (07/08/2013)
 * Copyright (c) 2012 Andrew Meyer
 * Licensed under the MIT License: http://en.wikipedia.org/wiki/MIT_License
 * Requires: jQuery v1.4+
 * Updated: 2018/02/03 steveb@nist.gov to match current jQuery plugin docs and to set defaults for NIST
*/

(function( $ ) {
	$.fn.leaveNotice = function(opt){
		
		// define default parameters
		var defaults = {
			siteName: 'NIST',
			exitMessage: '<h2><strong>Thank you for visiting {SITENAME}.</strong></h2><p>We hope your visit was informative.</p><p>We have provided this link to a non-NIST site because it has information that may be of interest to our users. NIST does not necessarily endorse the views expressed or the facts presented on this site. Further, NIST does not endorse any commercial products that may be advertised or available on this site.</p>',
			preLinkMessage: "<div class='setoff'><p>You will now be directed to:<br/>{URL}</p></div>",
			linkString: "", 
			newWindow: false,
			timeOut: 10000,
			overlayId: "ln-blackout",
			messageBoxId: "ln-messageBox",
			messageHolderId: "ln-messageHolder",
			linkId: "ln-link",
			displayUrlLength: 50,
			overlayAlpha: 0.3
		};
		
		var options = $.extend(defaults, opt);
		
		return this.each(function(){
			el = $(this);
			
			//URL the link goes to
			var url=el.attr('href');
			
			//Truncates long URLs to keep 'em pretty
			//Sets length to option value
			var ulen=options.displayUrlLength;
			//If the URL is longer than desired length, add an ellipsis
			if (url.length>=ulen) {
				var suffix = "...";
			} else {
				var suffix = "";	
			}
			//build short URL string
			var shortUrl=url.substr(0,ulen)+suffix;
			
			//Get "title" attribute of the link
			var title = el.attr('title');
			
			//Sets linkText to title if there is one. If not, it defaults to the URL
			if (title === undefined || title=="") {
				var linkText=shortUrl;
			} else {
				var linkText=shortUrl + title;
			}
			
			options.timeOut = (options.newWindow) ? 0 : options.timeOut;

			el.click(function(){
				//Append overlay box
				$('body').append('<div id="' + options.overlayId + '"></div>');
				//Append message holder and message boxes
				$('body').append('<div id="' + options.messageHolderId + '"><div id="' + options.messageBoxId + '"></div></div>');
				//If not turned off in the options, set the opacity on the overlay box to the overlayAlpha option. This is the default while opacity is not supported as a CSS property in all the browsers. In the future, the opacity should be handled from the CSS.
				if (options.overlayAlpha!==false) {
					$('#'+options.overlayId).css('opacity',options.overlayAlpha);
				}
				//Put all the HTML together from the options and replace the keywords with their appropriate data.
				preFilteredContent=options.exitMessage + options.preLinkMessage;
				msgContent=preFilteredContent.replace(/\{URL\}/g, '<a id="'+options.linkId+'" href="'+url+'" title="'+url+'"'+options.linkString+'>'+linkText+'</a>');
				msgContent=msgContent.replace(/\{SITENAME\}/g, options.siteName);
				//If timer is enabled, add the close controls to the HTML
				if (options.timeOut>0) {
					msgContent+='<p id="ln-cancelMessage"><a href="#close" id="ln-cancelLink">Cancel</a> or press the ESC key.</p>';
				} else {
					msgContent+='<p id="ln-cancelMessage">Click the link above to continue or <a href="#close" id="ln-cancelLink">Cancel</a></p>';
				}

				//Append the HTML to the message box
				$('#'+options.messageBoxId).append(msgContent);
				
				//If the timer is enabled, set the timer to follow link after desired time.
				if (options.timeOut>0) {
					leaveIn=setTimeout(function(){
						$('#ln-cancelMessage').html('<em>Loading...</em>');
						window.location.href=url;
					},options.timeOut);
				} else {
					leaveIn=false;
				}
				
				//if newWindow is turned on, add target and click behavior
				if (options.newWindow) {
					$('a#'+options.linkId).attr('target', '_blank').click(function(){
						closeDialog(options, leaveIn);
					})
				}

				//Apply event handler to pressing the close link
				$('#ln-cancelLink').click(function(){
					closeDialog(options, leaveIn);
					return false;
				});
				
				//Set up event handler for the ESC key
				$(document).bind('keyup', function(e){
					if (e.which==27) {
						closeDialog(options, leaveIn);
					}
				});
				
				// Clears the display when leaving the page to prevent it from showing upon returning.
				$(window).on('unload', function(){
					closeDialog(options, leaveIn);
				});
				
				return false;
			});
		});
	};
	
	// private function to close the dialog. This may be public in future
	// releases, but for now it has to be private.
	function closeDialog(options, timer) {
		if (options.timeOut>0) {
			clearTimeout(timer);
		}
		$('#'+options.overlayId+', #'+options.messageHolderId).fadeOut('fast',function(){
			$('#'+options.overlayId+', #'+options.messageHolderId).remove();
		});
		$(document).unbind('keyup');
	}
	
	// end and return jQuery object
})( jQuery );