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

    var HistogramData = Backbone.Model.extend({
        urlRoot: '/hist',
    });

    var ChartView = Backbone.View.extend({
        initialize: function() {
            this.model.bind('sync', this.render, this);
        },

        render: function() {
            histData = this.model.toJSON();
            this.$el.html(histData.col_name)
            return this;
        },
    })

    var columnsList = new ColumnList();
    var columnsListView = new ColumnListView({el: $("#column-select"), collection: columnsList});

    var histogramData = new HistogramData();
    var chartView = new ChartView({el: '#chart', model: histogramData});

    columnsList.fetch(); // Start calling at the end
})
