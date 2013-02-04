$(function() {
    var HistogramData = Backbone.Model.extend({
        urlRoot: '/hist',
    });

    var ChartView = Backbone.View.extend({
        initialize: function() {
            this.model.bind('sync', this.render, this);
        },

        render: function() {
            this.$el.html('');
            histogramData = this.model.toJSON();

            var width = 960;
            var height = 500;
            var padding = {top: 15, right: 15, bottom: 25, left: 30};

            var xScale = d3.scale.linear()
                    .domain([histogramData.x_min, histogramData.x_max])
                    .range([padding.left, width - padding.right]);
            var yScale = d3.scale.linear()
                    .domain([0, histogramData.y_max])
                    .range([height - padding.bottom, padding.top]);

            var values = histogramData.values
            var data = d3.layout.histogram()
                .bins(20)
                (values);
            var barWidth = xScale(data[1].x) - xScale(data[0].x) - 1

            var svg = d3.select(this.el)
                .append("svg")
                    .attr("width", width)
                    .attr("height", height)

            if (histogramData.nans > 0){
                xScale = d3.scale.linear()
                    .domain([histogramData.x_min, histogramData.x_max])
                    .range([padding.left + barWidth, width - padding.right]);
                barWidth = (xScale(data[1].x) - xScale(data[0].x)) - 1

                yNans = histogramData.nans
                svg.append("g")
                    .attr("class", "bar-nans")
                    .attr("transform", function(d) { return "translate(" + (padding.left + 2) + "," + yScale(yNans) + ")"; })
                    .append("rect")
                        .attr("width", barWidth - 1)
                        .attr("height", height - yScale(yNans) - padding.bottom);
            }

            var bar = svg.selectAll(".bar")
                .data(data)
              .enter().append("g")
                .attr("class", "bar")
                .attr("transform", function(d) { return "translate(" + xScale(d.x) + ", " + yScale(d.y) + ")"});

            bar.append("rect")
                .attr("width", barWidth)
                .attr("height", function(d) { return height - yScale(d.y) - padding.bottom; });

            // Axis
            var xAxis = d3.svg.axis()
                .scale(xScale)
                .orient("bottom");

            var yAxis = d3.svg.axis()
                  .scale(yScale)
                  .orient("left")
                  .ticks(5);

            svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + (height - padding.bottom) + ")")
                .call(xAxis);
            svg.append("g")
                .attr("class", "y axis")
                .attr("transform", "translate(" + padding.left + ",0)")
                .call(yAxis);

            return this;
        },
    })

    var histogramData = new HistogramData();
    var chartView = new ChartView({el: '#chart', model: histogramData});
    histogramData.set('id', 1) // Testing
    histogramData.fetch(); // Testing
})
