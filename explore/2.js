$(function(){

    var HistogramData = Backbone.Model.extend({
        urlRoot: '/hist',
    });

    var ChartView = Backbone.View.extend({
        initialize: function() {
            this.model.bind('sync', this.render, this);
        },

        render: function() {
            histData = this.model.toJSON();
            this.$el.html(histData.col_name);
            return this;
        },
    })

    var histogramData = new HistogramData();
    var chartView = new ChartView({el: '#chart', model: histogramData});
    histogramData.set('id', 1)
    histogramData.fetch()
})
