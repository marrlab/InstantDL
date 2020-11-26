#include <type_traits>
#include "CompoundProvider.hpp"

// inserts a new element in the inheritance chain, if it wasn't contained yet
template <typename CompoundProviderType, typename Element, bool IsContained = std::is_base_of<Element, CompoundProviderType>::value>
struct Insert {
	// Element is already contained
	typedef CompoundProviderType Value;
};
template <typename ... T, typename Element>
struct Insert<CompoundProvider<T...>, Element, false> {
	// add Element to type chain
	typedef CompoundProvider<T..., Element> Value;
};

/**
 * Template meta-function to merge two statistics providers. The resulting type 
 * inherits from each unique provider exactly once.
 */
// two different elements
template <typename Element1, typename Element2>
struct MergeProviders {
	typedef CompoundProvider<Element1, Element2> Value;
};
// two elements that are the same
template <typename Element>
struct MergeProviders<Element, Element> {
	typedef Element Value;
};
// first is a type chain
template <typename ... T, typename Element>
struct MergeProviders<CompoundProvider<T...>, Element> {
	typedef typename Insert<CompoundProvider<T...>, Element>::Value Value;
};
// second is a type chain
template <typename ... T, typename Element>
struct MergeProviders<Element, CompoundProvider<T...>> {
	typedef typename Insert<CompoundProvider<T...>, Element>::Value Value;
};
// both are type chains -> iterate
template <typename ... T, typename ... S>
struct MergeProviders<CompoundProvider<T...>, CompoundProvider<S...>> :
		public
				MergeProviders<
						typename MergeProviders<
							CompoundProvider<T...>,
							typename CompoundProvider<S...>::HeadType
						>::Value,
						typename CompoundProvider<S...>::Parent> {
};
// end of iteration
template <typename ... T>
struct MergeProviders<CompoundProvider<T...>, EndOfCompound> {
	typedef CompoundProvider<T...> Value;
};
