$(document).ready(function(){
    $("#submit").click(function(event){
        var uInput = $("#user-input").val();
        $.ajax({
              type: "POST",
              url: '/learning',
              data: JSON.stringify({userInput: uInput}),
              contentType: 'application/json',
              success: function(response){
                   $("#results").text(response.results);
                },
          });
    });
});
