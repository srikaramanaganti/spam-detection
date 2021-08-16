$('document').ready(function(){
    $("#but").click(function(){
        var msg = $('#input').val();
        if (msg =='')
        {
            $("#myModal").modal('show');
        }
        else{
        $.ajax({
			url: '/result',
			type: 'POST',
            data: {msg:msg},
			success: function(response){
                if(response === 'Spam')
                {
                    $("#div1").addClass("alert-danger");
                    $("#res").text(response);
                }
                if(response === 'Non-Spam'){
                    $("#div1").addClass("alert-success");
                    $("#res").text(response);
                }
                $("#div1").fadeIn(1000);
				$("#res").val(response);
			},
			error: function(error){
				console.log(error);
            }
		});
    }
    });
    $("#close").click(function(){
        var classes = $('#div1').prop('classList');
        // console.log(classes);
        //console.log(classes[1]);
        $("#div1").fadeOut(1000);
        setTimeout(function(){$("#div1").removeClass(classes[1])},1000);
        $('#input').val('');

    });

});