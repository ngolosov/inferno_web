<!DOCTYPE html>
<!--[if lt IE 7 ]><html class="ie ie6" lang="en"> <![endif]-->
<!--[if IE 7 ]><html class="ie ie7" lang="en"> <![endif]-->
<!--[if IE 8 ]><html class="ie ie8" lang="en"> <![endif]-->
<!--[if (gte IE 9)|!(IE)]><!--><html lang="en"> <!--<![endif]-->
<head>
<script src="https://unpkg.com/js-year-calendar@latest/dist/js-year-calendar.min.js"></script>
<script src="https://code.jquery.com/jquery-3.5.1.min.js" integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0=" crossorigin="anonymous"></script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/maphilight/1.4.0/jquery.maphilight.min.js"></script>
<script src="dates.js"></script>
<script>
$(function() {
    $('.maparea').maphilight();
});
</script>
<link rel="stylesheet" type="text/css" href="https://unpkg.com/js-year-calendar@latest/dist/js-year-calendar.min.css" />

<!--#include virtual="/head.ssi" -->

</head>
<body>

<!--#include virtual="/header.ssi" -->
<!--#include virtual="/navigation.ssi" -->

<div class="band">
    <div class="container">	
		<div class="one.column">

<center>
<h1>Infrared and visible images on {{ date }}</h1>

<p>
<input type="button" value="<< Previous day" onclick="javascript:window.open('https://prosecco.geog.psu.edu/inferno_web/{{ prev_date }}','_self')">
<input type="button" value="Calendar" onclick="javascript:window.open('https://prosecco.geog.psu.edu/inferno_web/','_self')"> 
<input type="button" value="Next day >>" onclick="javascript:window.open('https://prosecco.geog.psu.edu/inferno_web/{{ next_date }}','_self')">
</p>

<table>
  <tr>
    
    <th>Infrared image</th>
	<th>Visible image</th>

  </tr>

{% for image in images %}

<tr>
    <td> <center>{{ image[0][11:-4].replace("-", ":") }} </center></td>
	
	<td> <center> {{ image[1][11:-4].replace("-", ":") }} </center></td>
	
</tr>

<tr>
    <td> <img src = "/inferno_web/{{ date }}/IR_thumb/{{ image[0][:-4] }}.jpg" </img> </td>
	
	<td> <img src = "/inferno_web/{{ date }}/visible_thumb/{{ image[1][:-4] }}_thumb.jpg" </td>
	
</tr>


<tr>
    <td><center> <a href = "/inferno_web/{{ date }}/IR/{{ image[0][:-4] }}.seq">Download .seq file</a> &nbsp;/ &nbsp;
    <a href = "/inferno_web/{{ date }}/IR_csv/{{ image[0][:-4] }}.csv">Download .csv file</a> </center>
     </td>
	
		
    <td> <center><a href = "/inferno_web/{{ date }}/visible/{{ image[1][:-4] }}.jpg">View full-size JPG image</a></center></td>
</tr>

<tr>
    <td> &nbsp; </td>
	
	<td> &nbsp;</td>
	
</tr>

{% endfor %}

</table>

<h2>Image predictions:</h2>
<p>This section displays average temperatures for the next day, predicted by the convolution LSTM model. The graph below shows the mean temperature of the subset of the building wall for the today and next day predicted images</p>
<img src = "/inferno_web/{{ date }}/predictions_lstm/predictions_graph.png"> 
<br/>
<p>This section displays images for the next 24 hours. Click the thumbnail to download the predicted image in CSV format.</p>

<img src = "/inferno_web/{{ date }}/predictions_lstm/predictions_images.png" usemap="#image-map" class="maparea"> 


<map name="image-map">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_25.csv" coords="27,36,190,199" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_26.csv" coords="385,199,220,36" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_27.csv" coords="579,198,418,35" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_28.csv" coords="612,38,774,197" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_29.csv" coords="28,233,189,393" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_30.csv" coords="223,233,384,393" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_31.csv" coords="417,232,581,394" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_32.csv" coords="614,233,774,393" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_33.csv" coords="27,431,190,590" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_34.csv" coords="222,430,385,592" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_35.csv" coords="418,430,579,591" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_36.csv" coords="613,432,774,591" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_37.csv" coords="27,629,190,790" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_38.csv" coords="221,628,382,790" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_39.csv" coords="416,625,580,792" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_40.csv" coords="612,631,775,788" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_41.csv" coords="26,826,187,985" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_42.csv" coords="221,825,386,987" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_43.csv" coords="418,826,581,988" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_44.csv" coords="612,824,777,986" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_45.csv" coords="27,1022,188,1184" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_46.csv" coords="221,1022,384,1185" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_47.csv" coords="417,1023,580,1186" shape="rect">
    <area target="" alt="" title="" href="/inferno_web/{{ date }}/predictions_lstm/pred_48.csv" coords="612,1023,776,1183" shape="rect">
</map>
<center>
        <h4>Contact us about this project:</h4>
        <ul class="square">
          <li><strong>e-mail</strong>: <a href="mailto:golosov@psu.edu">golosov@psu.edu</a>
        </ul>
	</div> <!-- container -->
</div> <!-- band -->

<!--#include virtual="/footer.ssi" -->

<!-- End Document
================================================== -->
</body>
</html>
