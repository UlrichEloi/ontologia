$(document).ready(function(){
    //alert("thank you");

    //$('ul li:first').removeClass("active");
    
/* activer le lien quand on clique(MENU) */
    $('ul li a').click(function (){
      $('li a').removeClass("active");
      $(this).addClass("active");
    });
});
