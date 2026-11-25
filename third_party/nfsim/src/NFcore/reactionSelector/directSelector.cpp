/*
 * DirectSelector.cpp
 *
 *  Created on: Jul 23, 2009
 *      Author: msneddon
 */



#include "reactionSelector.hh"

#include <cmath>

using namespace std;
using namespace NFcore;


/* update() keeps Atot as a running sum, Atot - oldA + newA. Each update rounds
 * it twice, each time by at most about half an ulp of the largest value the sum
 * has held since it was last summed, so n updates leave it off by up to about
 * n * 2^-52 of that peak. A propensity added to a large total is rounded to the
 * total's spacing, and is not given back when the large term is taken off
 * again. A slow channel at 1e-6 beside a fast one at 1e11 was lost outright
 * once the fast one was 0, and beside one at 1e10 it came back as 1.907e-6
 * (bngsim GH #1666, RuleWorld/nfsim#123). The sum goes astray without a fall
 * too, where the roundings fall the same way every time: one molecule flipping
 * between two reactions beside a fixed background lost 2.2e-16 a cycle, and the
 * reaction listed last loses from every pick what the sum has lost.
 *
 * So Atot is summed afresh once it has fallen below 2^-10 of that peak, or
 * below 0, or once it has taken 2^12 updates since it was last summed, as
 * RuleMonkey sums its own every 4096 events. Where it is read, it has then
 * taken at most 2^12 updates more than one firing makes. With every rounding
 * the same way, the case that drifts, that leaves it off by at most about
 * 2^-40 of its peak. That is 2^-40 of a total near its peak, well below the
 * 2^-32 of the total to which a pick is drawn (NfsimRNG::random), and 2^-30 of
 * one that sits at 2^-10 of its peak. Where the roundings fall either way, it
 * is about 2^-46 and 2^-36.
 *
 * update() only counts and keeps the peak. The tests are made where the total
 * is read: before a waiting time is drawn from it (getAtotForDraw()) and before
 * a pick. Within a firing the total can dip and come back, as when a molecule
 * that carries nearly all of it leaves one reaction before it joins the next. A
 * sum is a pass over every reaction, where a pick stops at the one it draws, so
 * a test in update() would cost such a model a pass per firing. Where the total
 * is read, a sum comes once per 2^12 updates and once per fall by a factor of
 * 1024. Each is a pass over all N reactions, so a model whose total keeps
 * falling that far below its peak slows in proportion to N. One A flipping at
 * 1e6 and back at 1e-3, so that nearly every other firing is such a fall, ran
 * 1.8 times as long beside 1000 births at 1e-9 as without the tests, and 17
 * times as long beside 10,000. The count alone costs a pass per 4096 updates.
 *
 * RuleWorld/nfsim#128, merged upstream as 7501f675, makes the same test of the
 * fall, before a waiting time is drawn, and has no count.
 *
 * The sum reads each reaction's get_a(), the propensities the running sum is
 * made of, rather than computing every rate again as refactorPropensities()
 * does.
 *
 * A total that is not finite is never summed afresh here: it is the model's
 * own, and the host reports it. */
static const double ATOT_RESUM_FRACTION = 1.0 / 1024.0;   // 2^-10
static const unsigned int ATOT_RESUM_UPDATES = 4096;      // 2^12


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
	updatesSinceSum = 0;
}


void DirectSelector::resumIfStale()
{
	if(!std::isfinite(Atot)) return;
	if(Atot < AtotResumBelow || updatesSinceSum >= ATOT_RESUM_UPDATES) resumPropensities();
}


double DirectSelector::getAtotForDraw()
{
	resumIfStale();
	return Atot;
}


double DirectSelector::update(ReactionClass *r,double oldA, double newA)
{
	Atot-=oldA;
	Atot+=newA;
	++updatesSinceSum;
	if(Atot > AtotPeak) {
		AtotPeak = Atot;
		AtotResumBelow = AtotPeak * ATOT_RESUM_FRACTION;
	}
	return Atot;
}



double DirectSelector::getNextReactionClass(ReactionClass *&rc)
{
	resumIfStale();
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

