[//]: # (author: samtenka)
[//]: # (change: 2019-06-10)
[//]: # (create: 2019-06-10)
[//]: # (descrp: introduction to project goals and code organization)

# self-reg
#### with [Dan A Roberts](http://www.sns.ias.edu/~roberts/) and [Sho Yaida](http://sites.google.com/site/shoyaida/)
#### understanding and improving gradient-based optimization by perturbing away from ODE

## project goals 

## implementation 

### conventions of terminology

Our mathematical analysis works with tensor fields, tensor stalks, and tensors, each random or 
certain.  **Fields** may be evaluated at any theta, **stalks** at theta0 may be evaluated at
any theta infinitesimally close to theta0, and **ordinary tensors** at theta0 live solely in
(powers of) theta0's tangent space.  Their **random** versions depend on data while their
deterministic or **certain** versions, for instance expectations over the whole population or
samples at fixed data, do not.  In our tensorflow implementation, the corresponding types are: 

    MATHEMATICAL OBJECT                             PYTHON TYPE
    
    weight x random  tensor field           <--->   PointedLandscape 
             random  tensor stalk x datapt   --->   torch Tensor 
             random  tensor       x datapt   --->   numpy array 
             certain tensor field           <---    PointedLandscape x datapt
             certain tensor stalk           <--->   torch Tensor     
             certain tensor                 <--->   numpy array 

### directory structure
    self-reg/
        README.md
        experiment/
            Makefile
            derivator.py                differentiate loss landscape object, writing to gradstats object
            gradstats.py                define log type for gradient statistics  
            landscape.py                define loss landscape type
            optimlogs.py                define log type for descent trajectory summaries
            predictor.py                predict descent trajectory given gradient statistics  
            simulator.py                implement SGD on loss landscape object, writing to optimlogs object 
            visualize.py                plot predictor's predictions vs optimlogs, writing to images/plots/
        images/
            diagrams/
            plots/
        writeups/
            Makefile
            perturb.tex 
