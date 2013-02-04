$(function(){

    var HistogramData = Backbone.Model.extend({
        urlRoot: '/hist',
    });

    var histogramData = new HistogramData({id: 1})
    histogramData.fetch({
        success: function (response) {
            data = response.toJSON()
            alert(data.col_name)
        }
    })
})
