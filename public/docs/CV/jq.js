$('a.links').click(function (e){
   e.preventDefault();
   var div_id = $('a.links').index($(this))
   $('.divs').hide().eq(div_id).show();
});