/**
 * This file contains the command line interface for the decision jungle library. It essentially let's you train a model
 * on some dataset provided by a data file, save the model to a model file and predict new values from a data file
 * based on a learned model from a model file. Please see the help option for further information. 
 * 
 * @author Tobias Pohlen <tobias.pohlen@rwth-aachen.de>
 * @version 1.0
 */
#ifndef CLIINTERFACE_H
#define	CLIINTERFACE_H

#include <memory>
#include <map>
#include <vector>
#include <string>
#include "misc.h"
#include "config.h"

namespace decision_jungle {
    /**
     * Forward declarations
     */
    class AbstractCLIFunction;
    class ParameterBag;
    
    /**
     * This exception is thrown when a cli function cannot be found.
     */
    DEFINE_EXCEPTION(CLIFunctionNotFoundException)
    
            
    /**
     * Creates a new instance of T
     * 
     * @return new Instance
     */
    template<typename T>
    AbstractCLIFunction * createT()
    {
        return new T;
    }
    
    /**
     * This class holds all parameters provided to the command line tool. It parses the parameters as follows:
     * -{text} {value} will be saved as a setting parameter without any contextual relation
     * {text} will be saved in a list of ordered arguments. 
     * 
     * Example
     * $ cli function -t=5 -e=3 -test="this is a test" -d model.file data.file
     * This will be parsed as:
     * Parameters:
     * - t -> 5
     * - e -> 3
     * - test -> this is a test
     * - d -> 1
     * Arguments
     * 1) function
     * 2) model.file
     * 3) data.file
     */
    class ArgumentBag {
    private:
        /**
         * These are the provided parameters
         */
        std::map<std::string, std::string> parameters;
        /**
         * These are the provided ordered arguments
         */
        std::vector<std::string> arguments;
        
    public:
        typedef ArgumentBag self;
        typedef std::shared_ptr<self> ptr;
        
        /**
         * Returns the parameter bag
         * 
         * @return parameters
         */
        std::map<std::string, std::string> & getParameters()
        {
            return parameters;
        }
        
        /**
         * Returns the argument bag
         * 
         * @return arguments
         */
        std::vector<std::string> & getArguments()
        {
            return arguments;
        }
        
        /**
         * This is a factory for the argument bags
         */
        class Factory {
        public:
            /**
             * Creates a blank argument bag
             * 
             * @return blank argument bag
             */
            static ArgumentBag::ptr create()
            {
                return ArgumentBag::ptr(new ArgumentBag);
            }
            
            /**
             * Creates an argument bag from the cli arguments
             * 
             * @param start The number to start parsing from
             * @param number of provided arguments
             * @param argument array
             */
            static ArgumentBag::ptr createFromCLIArguments(const int start, const int argc, const char** argv);
        };
    };
    
    /**
     * This is the base class for all command line functionality. 
     */
    class AbstractCLIFunction {
    private:
        /**
         * These are the parameters provided as arguments to the command line
         */
        ArgumentBag::ptr arguments;
        
    protected:
        /**
         * Returns the argument bag
         * 
         * @return Argument bag
         */
        ArgumentBag::ptr getArguments()
        {
            return arguments;
        }
        
    public:
        typedef AbstractCLIFunction self;
        typedef std::shared_ptr<self> ptr;
        
        /**
         * Executes the command/function
         */
        virtual int execute() = 0;
        
        /**
         * Returns the help documentation of the function
         */
        virtual const char* help() = 0;
        
        /**
         * Returns the short help text for the overview
         */
        virtual const char* shortHelp() = 0;
        
        /**
         * This is the factory for the cli functions
         */
        class Factory {
        public:
            /**
             * This function holds the references to all available cli functions
             */
            typedef std::map< std::string, AbstractCLIFunction*(*)() > ClassMap;
            
        private:
            /**
             * All registered functions
             */
            static ClassMap functionMap;
            
        protected:
            /**
             * Returns the function class map
             * 
             * @return function map
             */
            static ClassMap & getFunctionMap()
            {
                return functionMap;
            }
            
        public:
            /**
             * Creates a command line function from a provided argument bag
             * 
             * @param _argumentBag The provided cli arguments
             * @return A new function instance
             */
            static AbstractCLIFunction::ptr createFromArgumentBag(ArgumentBag::ptr _argumentBag) throw(CLIFunctionNotFoundException);
        };
            
        /**
         * This class is used in order to register a new cli function. Simply add it as a private
         * member of your function class
         */
        template<typename T>
        class RegisterFunction : public AbstractCLIFunction::Factory {
        public:
            /**
             * Registers the function
             * 
             * @param s the name under which is will be registered
             */
            RegisterFunction(std::string const& s) {
                getFunctionMap()[s] = &createT<T>;
            }
        };
    friend class AbstractCLIFunction::Factory;
    };
    
    /**
     * This function displays the help message. 
     */
    class HelpCLIFunction : public AbstractCLIFunction {
    private:
        /**
         * This is needed in order to register the function
         */
        static AbstractCLIFunction::RegisterFunction<HelpCLIFunction> reg;
        
    public:
        /**
         * Executes the command/function
         */
        virtual int execute();
        
        /**
         * Returns the help documentation of the function
         */
        virtual const char* help();
        
        /**
         * Returns the short help text for the overview
         */
        virtual const char* shortHelp();
    };
    
    /**
     * This function let's you classify known and unknown data
     */
    class ClassifyCLIFunction : public AbstractCLIFunction {
    private:
        /**
         * This is needed in order to register the function
         */
        static AbstractCLIFunction::RegisterFunction<ClassifyCLIFunction> reg;
        
    public:
        /**
         * Executes the command/function
         */
        virtual int execute();
        
        /**
         * Returns the help documentation of the function
         */
        virtual const char* help();
        
        /**
         * Returns the short help text for the overview
         */
        virtual const char* shortHelp();
    };
    
    /**
     * This function lets you learn a new classifier from known data
     */
    class TrainCLIFunction : public AbstractCLIFunction {
    private:
        /**
         * This is needed in order to register the function
         */
        static AbstractCLIFunction::RegisterFunction<TrainCLIFunction> reg;
        
    public:
        /**
         * Executes the command/function
         */
        virtual int execute();
        
        /**
         * Returns the help documentation of the function
         */
        virtual const char* help();
        
        /**
         * Returns the short help text for the overview
         */
        virtual const char* shortHelp();
    };
    
    /**
     * This function displays the current version of the library
     */
    class VersionCLIFunction : public AbstractCLIFunction {
    private:
        /**
         * This is needed in order to register the function
         */
        static AbstractCLIFunction::RegisterFunction<VersionCLIFunction> reg;
        
    public:
        /**
         * Executes the command/function
         */
        virtual int execute();
        
        /**
         * Returns the help documentation of the function
         */
        virtual const char* help();
        
        /**
         * Returns the short help text for the overview
         */
        virtual const char* shortHelp();
    };
}

#endif

