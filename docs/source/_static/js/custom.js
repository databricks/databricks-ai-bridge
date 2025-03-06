/* This adds target='_blank' to all external links,
   allowing for them to open in a new browser window/tab. */
$(document).ready(function () {
    $('a.external').attr('target', '_blank');
});
