#ifndef MISC_HPP
#define	MISC_HPP

#include <exception>
#define INC_DEBUG (decision_jungle::__debugCounter++);
#define DEC_DEBUG (decision_jungle::__debugCounter--);
#define INC_DEBUG2 (decision_jungle::__debugCount++);
#define DEC_DEBUG2 (decision_jungle::__debugCount--);


/**
 * This file contains miscellaneous functions and definitions needed allover the place. 
 * 
 * @author Tobias Pohlen
 * @version 1.0
 */

/**
 * Deletes a pointer if and only if it is not null
 */
#define DELETE_PTR(p) if((p) != 0) { delete (p); }

/**
 * This makro helps you creating custom exception classes which accept error messages as constructor arguments. 
 * You can define a new exception class by: DEFINE_EXCEPTION(classname)
 * You can throw a new exception by: throw classname("Error message");
 */
#define DEFINE_EXCEPTION(classname)		\
	class classname : public std::exception {	\
	public:		\
		classname() { this->ptrMessage = 0; };	\
		classname(const char* _ptrMessage) : ptrMessage(_ptrMessage) {};	\
		classname(std::string str) {_msg = str; ptrMessage = _msg.c_str(); };	\
		classname(const char* _ptrMessage, int l) : ptrMessage(_ptrMessage) { };	\
        virtual ~classname() throw() {}; \
		virtual const char* what() const throw() { return (this->ptrMessage != 0 ? this->ptrMessage : "No Message"); }	\
	private: \
        std::string _msg; \
		const char* ptrMessage;		\
	}; 

/**
 * Defines the pointer types for a class. i.e. the shared_ptr and the self reference. 
 */
#define DEFINE_CLASS_PTR(classname) typedef classname self;	typedef std::shared_ptr<self> ptr;

#endif

