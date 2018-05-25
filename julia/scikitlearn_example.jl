cd(dirname(@__FILE__))

using DataFrames
using CSV

using ScikitLearn

using StatPlots
using PlotThemes

theme(:gluon_report) ## theme for the plot package

D_full = CSV.read("creditcard.csv", types=Dict("Time"=>Float64), rows=20001, allowmissing=:none) ## load data ,
D_full[:V0] = fill(1.0, length(D_full[:V1]))
D_full[D_full[:Class] .== 0.0, :Class] = -1.0
N_full = size(D_full, 1)

srand(1) ## set the random seed

@show N_train = (Int ∘ ceil)(N_full * 0.6)
@show N_test = N_full - N_train

train_id = shuffle!([fill(true, N_train) ; fill(false, N_test)])
@show train_id[1:10]

### train set
X_train = Array(D_full[train_id, [[Symbol("V", i) for i in 0:28]; :Amount] ])
y_train = Array(D_full[train_id, :Class])

### test set
X_test  = Array(D_full[!train_id, [[Symbol("V", i) for i in 0:28]; :Amount] ])
y_test = Array(D_full[!train_id, :Class])


@sk_import linear_model: LogisticRegression
@sk_import linear_model: RidgeClassifier
@sk_import ensemble: RandomForestClassifier
import ScikitLearn: CrossValidation.cross_val_score

cvs_logreg = cross_val_score(LogisticRegression(fit_intercept=true), X_train, y_train; cv=3)
cvs_ridge  = cross_val_score(RidgeClassifier(), X_train, y_train; cv=3)
cvs_ranfor = cross_val_score(RandomForestClassifier(), X_train, y_train; cv=3)
println("mean cvs for logistic regression : $(mean(cvs_logreg))")
println("mean cvs for ridge classifier    : $(mean(cvs_ridge))")
println("mean cvs for random forest class : $(mean(cvs_ranfor))")

x_val = :V2
y_val = :V4
scatter([D_full[D_full[:Class] .== -1.0, x_val], D_full[D_full[:Class] .== 1.0, x_val]], [D_full[D_full[:Class] .== -1.0, y_val], D_full[D_full[:Class] .== 1.0, y_val]], label=["-1", "1"], xlabel="$x_val", ylabel="$y_val" )


@df D_full corrplot([:V2 :V3 :V4])
