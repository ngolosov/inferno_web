var dates = [
    {% for date in dates %}
	{ startDate: new Date({{ date[0] }}, {{ date[1] }}-1, {{ date[2] }}), endDate: new Date({{ date[0] }}, {{ date[1] }}-1, {{ date[2] }}),  color: "orange" },
	{% endfor %}
  ];