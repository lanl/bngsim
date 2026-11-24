/*
 * DirectSelector.cpp
 *
 *  Created on: Jul 23, 2009
 *      Author: msneddon
 */



#include "reactionSelector.hh"

using namespace std;
using namespace NFcore;


/* update() keeps Atot as a running sum, Atot - oldA + newA. Each update rounds
 * it twice, each time by at most about half an ulp of the largest value the sum
 * has held since it was last summed, so n updates leave it off by up to about
 * n * 2^-52 of that peak. A propensity added to a large total is rounded to the
 * total's spacing, and is not given back when the large term is taken off
 * again. A slow channel at 1e-6 beside a fast one at 1e11 was lost outright
 * once the fast one was 0, and beside one at 1e10 it came back as 1.907e-6
 * (bngsim GH #1666).
 *
 * So Atot is summed afresh once it falls below 2^-10 of that peak, or below 0.
 * Above that, n updates cost it at most n * 2^-42 of its own value, and more
 * nearly sqrt(n) * 2^-42, since roundings fall either way. That stays below the
 * 2^-32 of the total to which a pick is drawn (NfsimRNG::random) for 1024
 * updates at the worst and about a million as a rule. A sum costs the one pass
 * over the reactions that every pick makes, and resets the peak, so it comes at
 * most once per fall by a factor of 1024, and never in a run whose total does
 * not fall that far.
 *
 * The sum reads each reaction's get_a(), not update_a(): an update can come in
 * the middle of a firing, where a rate that reads an observable is held and
 * must not be computed (RuleWorld/nfsim#115). A held reaction's get_a() is the
 * propensity the sum has for it, and the update that ends the hold takes it
 * off again.
 *
 * A total that is not a number fails both tests and is kept: it is the model's
 * own, and the host reports it. */
static const double ATOT_RESUM_FRACTION = 1.0 / 1024.0;   // 2^-10


DirectSelector::DirectSelector(vector <ReactionClass *> &rxns, System *sys) :
	ReactionSelector(sys)
{
	this->Atot = 0;
	this->n_reactions = rxns.size();
	this->reactionClassList = new ReactionClass *[n_reactions];
	for(int r=0; r<n_reactions; r++) {
		reactionClassList[r] = rxns.at(r);
		Atot += reactionClassList[r]->get_a();
	}
	markSummed();
}



DirectSelector::~DirectSelector()
{
	Atot = 0;
	n_reactions = 0;
	delete [] reactionClassList;
}

double DirectSelector::refactorPropensities()
{
	Atot = 0;
	for(int r=0; r<n_reactions; r++) {
		Atot += reactionClassList[r]->update_a();
	}
	markSummed();
	return Atot;
}


void DirectSelector::resumPropensities()
{
	Atot = 0;
	for(int r=0; r<n_reactions; r++) {
		Atot += reactionClassList[r]->get_a();
	}
	markSummed();
}


void DirectSelector::markSummed()
{
	AtotPeak = Atot > 0 ? Atot : 0;
	AtotResumBelow = AtotPeak * ATOT_RESUM_FRACTION;
}


double DirectSelector::update(ReactionClass *r,double oldA, double newA)
{
	Atot-=oldA;
	Atot+=newA;
	if(Atot > AtotPeak) {
		AtotPeak = Atot;
		AtotResumBelow = AtotPeak * ATOT_RESUM_FRACTION;
	} else if(Atot < AtotResumBelow) {
		resumPropensities();
	}
	return Atot;
}



double DirectSelector::getNextReactionClass(ReactionClass *&rc)
{
	// Nothing fires at a total of 0, or one that is not a number. random(0) is
	// 0, which the first reaction would pass below with a propensity of 0; a
	// pick that passed every reaction sums them afresh and comes back here.
	if(!(Atot > 0)) {
		rc = 0;
		return 0;
	}
	double randNum = sys_->getRNG().random(Atot);

	double a_sum=0, last_a_sum=0;

	//WARNING - DO NOT USE THE DEFAULT C++ RANDOM NUMBER GENERATOR FOR THIS STEP
	// - IT INTRODUCES SMALL NUMERICAL ERRORS CAUSING THE ORDER OF RXNS TO
	//   AFFECT SIMULATION RESULTS
	for(int r=0; r<n_reactions; r++) {
		a_sum += reactionClassList[r]->get_a();
		if(randNum <= a_sum)
		{
			rc = reactionClassList[r];
			return (randNum-last_a_sum);
		}
		last_a_sum = a_sum;
	}

	this->refactorPropensities();
	return getNextReactionClass(rc);

}


double DirectSelector::getAtot()
{
	return Atot;
}

