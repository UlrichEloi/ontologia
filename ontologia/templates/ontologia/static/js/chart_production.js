
    $(document).ready(function(){

       var endpoint = "/ontologia/api/chart/data/";
       customersDjango = {{ customers }}
       //console.log(customersDjango-1);
       //var endpoint = $("div").attr("url-endpoint");
       $.ajax({
         method:"GET",
         url:endpoint,
         success: function(data){
           labels = data.labels;
           defaultData = data.defaultData;
           setChart();
           //console.log(data);
           //console.log(data.customers * 234);
         },
         error: function(error_data){
           console.log("error");
           console.log(error_data);
         },
       });

       var defaultData = [];
       var labels = [];


      function setChart(){
        var ctx = document.getElementById("myChart").getContext('2d');
        var myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '# of Votes',
                    data: defaultData,
                    backgroundColor: [
                        //'rgba(255, 255, 255, 0.2)',
                        'rgba(194, 10, 10, 0.2)',
                        'rgba(194, 10, 10, 0.2)',
                        'rgba(194, 10, 10, 0.2)',
                        'rgba(194, 10, 10, 0.2)',
                        'rgba(194, 10, 10, 0.2)',
                        'rgba(194, 10, 10, 0.2)',
                        // 'rgba(255, 99, 132, 0.2)',
                        // 'rgba(255, 99, 132, 0.2)',
                        // 'rgba(255, 99, 132, 0.2)',
                        // 'rgba(255, 99, 132, 0.2)',
                        // 'rgba(255, 99, 132, 0.2)',

                        // 'rgba(54, 162, 235, 0.2)',
                        // 'rgba(255, 206, 86, 0.2)',
                        // 'rgba(75, 192, 192, 0.2)',
                        // 'rgba(153, 102, 255, 0.2)',
                        // 'rgba(255, 159, 64, 0.2)'
                    ],
                    borderColor: [
                      'rgba(194, 10, 10, 0.2)',
                      'rgba(194, 10, 10, 0.2)',
                      'rgba(194, 10, 10, 0.2)',
                      'rgba(194, 10, 10, 0.2)',
                      'rgba(194, 10, 10, 0.2)',
                      'rgba(194, 10, 10, 0.2)',
                        // 'rgba(255,99,132,1)',
                        // 'rgba(54, 162, 235, 1)',
                        // 'rgba(255, 206, 86, 1)',
                        // 'rgba(75, 192, 192, 1)',
                        // 'rgba(153, 102, 255, 1)',
                        // 'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                },
                {
                    label: '# of Votes2',
                    data: defaultData,
                    backgroundColor: [
                        //'rgba(255, 255, 255, 0.2)',
                        'rgba(3, 97, 204, 0.2)',
                        'rgba(3, 97, 204, 0.2)',
                        'rgba(3, 97, 204, 0.2)',
                        'rgba(3, 97, 204, 0.2)',
                        'rgba(3, 97, 204, 0.2)',
                        'rgba(3, 97, 204, 0.2)',
                        // 'rgba(25, 99, 132, 0.2)',
                        // 'rgba(25, 99, 132, 0.2)',
                        // 'rgba(25, 99, 132, 0.2)',
                        // 'rgba(25, 99, 132, 0.2)',
                        // 'rgba(25, 99, 132, 0.2)',

                        // 'rgba(54, 162, 235, 0.2)',
                        // 'rgba(255, 206, 86, 0.2)',
                        // 'rgba(75, 192, 192, 0.2)',
                        // 'rgba(153, 102, 255, 0.2)',
                        // 'rgba(255, 159, 64, 0.2)'
                    ],
                    borderColor: [
                      'rgba(3, 97, 204, 0.2)',
                      'rgba(3, 97, 204, 0.2)',
                      'rgba(3, 97, 204, 0.2)',
                      'rgba(3, 97, 204, 0.2)',
                      'rgba(3, 97, 204, 0.2)',
                      'rgba(3, 97, 204, 0.2)',
                        // 'rgba(255,99,132,1)',
                        // 'rgba(54, 162, 235, 1)',
                        // 'rgba(255, 206, 86, 1)',
                        // 'rgba(75, 192, 192, 1)',
                        // 'rgba(153, 102, 255, 1)',
                        // 'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }
              ]
            },
            options: {
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero:true
                        }
                    }]
                }
            }
        });

      }

    });
