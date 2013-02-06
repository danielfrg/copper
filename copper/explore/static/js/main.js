$(function(){

    var ColumnListItem = Backbone.Model.extend();
    var ColumnList = Backbone.Collection.extend({
        url: '/columns',
        model: ColumnListItem
    });

    var ColumnListView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this, 'addOne', 'addAll');
            this.collection.bind('reset', this.addAll);
        },

        addOne: function(column) {
            // column is a ColumnListItem model
            var variables = { id: column.get('id'), name: column.get('name') };
            var template = _.template( $("#option_template").html(), variables);
            $(this.el).append(template);
        },

        addAll: function() {
            this.collection.each(this.addOne); // For each item in the collection call addOne
            $(".chzn-select").chosen();
        },

        events: {
            "change": "changeSelectedItem"
        },

        changeSelectedItem: function(evt) {
            histogramData.set('id', $(this.el).val())
            histogramData.fetch()
        }
    })

    var NumberBins = Backbone.Model.extend({
        defaults: {
            bins: 20,
        }
    });
    var NumberBinsView = Backbone.View.extend({
        events: {
            "change": "changeValue"
        },
        changeValue: function( event ){
            this.model.set('bins', parseInt(this.$el.val()));
        }
    });

    var HistogramData = Backbone.Model.extend({
        urlRoot: '/hist',
    });

    var ChartView = Backbone.View.extend({
        initialize: function() {
            this.model.get('histogramData').bind('sync', this.render, this);
            this.model.get('numberBins').bind('change', this.render, this);
        },

        render: function() {
            this.$el.html('');
            var histData = this.model.get('histogramData').toJSON();
            if (histData.id == undefined) {
                return this;
            }
            var nbins = this.model.get('numberBins').toJSON().bins;

            var width = 960;
            var height = 500;
            var padding = {top: 15, right: 15, bottom: 25, left: 30};

            var values = histData.values;
            var data = d3.layout.histogram()
                .bins(nbins)
                (values);

            var xmin = d3.min(data, function(d) { return d.x; });
            var xmax = d3.max(data, function(d) { return d.x + d.dx; });
            var ymax = d3.max(data, function(d) { return d.y; });
            ymax = Math.max(histData.nans, ymax);
            var xScale = d3.scale.linear()
                    .domain([xmin, xmax])
                    .range([padding.left, width - padding.right]);
            var yScale = d3.scale.linear()
                    .domain([0, ymax])
                    .range([height - padding.bottom, padding.top]);

            var barWidth = xScale(data[1].x) - xScale(data[0].x) - 1

            var svg = d3.select(this.el)
                .append("svg")
                    .attr("width", width)
                    .attr("height", height)

            if (histData.nans > 0){
                xScale = d3.scale.linear()
                    .domain([xmin, xmax])
                    .range([padding.left + barWidth, width - padding.right]);
                barWidth = (xScale(data[1].x) - xScale(data[0].x)) - 1

                yNans = histData.nans
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
                .attr("class", "hint hint--top")
                .attr("data-hint", "hober me")
                .attr("data-placement", "top")
                .attr("data-content", "Vivamus sagittis lacus vel augue laoreet rutrum faucibus.")
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

    var ChartModel = Backbone.Model.extend();

    var columnsList = new ColumnList();
    var columnsListView = new ColumnListView({el: $("#column-select"), collection: columnsList});

    var numberBins = new NumberBins();
    var numberBinsView = new NumberBinsView({el: $("#num-bins"), model: numberBins});

    var histogramData = new HistogramData();
    var chartModel = new ChartModel({histogramData: histogramData, numberBins: numberBins});
    var chartView = new ChartView({el: '#chart', model: chartModel});

    columnsList.fetch();
})
